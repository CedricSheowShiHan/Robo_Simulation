"""
issue_card_demo.py  --  end-to-end reception demo on scene_dual.xml

Sequence:
  LEFT arm  = "identifier": moves to a look pose, points its RealSense/left_wrist_cam
              at the guest, renders frames, and a (stub) recogniser returns an identity.
  RIGHT arm = "issuer":     the identity maps to one staff card; the right arm reaches
              into the fan rack, grasps that card's top edge, lifts it out, and presents
              it over the counter to the guest, then opens and retracts.

Grasp model: a "kinematic grasp" -- when the gripper closes on the card we record the
card->gripper relative transform and rigidly follow the gripper each step until release.
This makes the pick 100% reliable regardless of finger/friction tuning. A physical
friction grasp or a <weld> equality is the more realistic alternative (see NOTE below).

Run:
    python3 issue_card_demo.py            # interactive viewer
    python3 issue_card_demo.py --video    # also write out/ third-person + wrist-cam mp4s
    python3 issue_card_demo.py --headless  # no viewer, just step (for CI / video)
"""

import argparse
import os
import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "scene_dual.xml")

# ----------------------------------------------------------------------------- #
#  small math helpers
# ----------------------------------------------------------------------------- #

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def look_rotation(z_dir, up_hint=(0.0, 0.0, 1.0)):
    """Rotation matrix whose local +z axis points along z_dir (world)."""
    z = normalize(np.asarray(z_dir, float))
    up = np.asarray(up_hint, float)
    if abs(np.dot(z, normalize(up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


TOOL_DOWN = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]])   # local +z -> world -z  (approach from above)


def orientation_error(R_cur, R_des):
    """Angular error vector (world) that rotates R_cur toward R_des."""
    return 0.5 * (np.cross(R_cur[:, 0], R_des[:, 0])
                  + np.cross(R_cur[:, 1], R_des[:, 1])
                  + np.cross(R_cur[:, 2], R_des[:, 2]))


# ----------------------------------------------------------------------------- #
#  model wrapper
# ----------------------------------------------------------------------------- #

class Scene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                    for i in range(m.nu)}

        # arm joint indices (6 DoF each) in qpos and dof/jac space
        self.arm = {
            "left":  dict(q=np.arange(0, 6),  dof=np.arange(0, 6),
                          act=[self.aid[f"left_joint{k}"] for k in range(1, 7)],
                          grip=self.aid["left_gripper"],
                          site=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")),
            "right": dict(q=np.arange(8, 14), dof=np.arange(8, 14),
                          act=[self.aid[f"right_joint{k}"] for k in range(1, 7)],
                          grip=self.aid["right_gripper"],
                          site=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")),
        }
        self.GRIP_OPEN = float(m.actuator_ctrlrange[self.aid["left_gripper"], 1])   # 0.035
        self.GRIP_SHUT = float(m.actuator_ctrlrange[self.aid["left_gripper"], 0])   # 0.0

        self.card_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"staff_card_{i}")
                         for i in range(1, 6)]
        self.card_qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in self.card_bid]
        self.card_dofadr = [m.jnt_dofadr[m.body_jntadr[b]] for b in self.card_bid]
        self.guest_head_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "guest_head")
        self.guest_hand_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "guest_hand")

        mujoco.mj_forward(self.model, self.data)

    # -- kinematics queries ------------------------------------------------- #
    def tcp_pose(self, arm):
        s = self.arm[arm]["site"]
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    def card_top(self, idx):
        """World point at the exposed top edge of staff_card_{idx+1}."""
        b = self.card_bid[idx]
        R = self.data.xmat[b].reshape(3, 3)
        return self.data.xpos[b] + R @ np.array([0.0, 0.0, 0.043])

    def guest_head(self):
        return self.data.geom_xpos[self.guest_head_gid].copy()

    def guest_hand(self):
        return self.data.site_xpos[self.guest_hand_sid].copy()

    # -- inverse kinematics -------------------------------------------------- #
    # Position-only damped-least-squares IK on the arm's 6 joints, always seeded
    # from a bent "ready" posture (joint2 has range [0, pi] -- seeding from all
    # zeros pins it on its lower limit and IK stalls). The Piper is small and
    # adding an orientation constraint makes these near-limit reaches infeasible,
    # so we let the tool take its natural orientation; the kinematic grasp does
    # not depend on it.
    IK_SEED = np.array([0.0, 1.4, -1.2, 0.0, 0.6, 0.0])

    def solve_ik(self, arm, target_pos, seed_arm=None, iters=120, pos_tol=1e-3, damp=1e-4):
        m, a = self.model, self.arm[arm]
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        d.qpos[a["q"]] = self.IK_SEED if seed_arm is None else seed_arm
        d.qvel[:] = 0
        qadr, dof, site = a["q"], a["dof"], a["site"]
        lo, hi = m.jnt_range[dof, 0], m.jnt_range[dof, 1]
        jp = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = target_pos - d.site_xpos[site]
            if np.linalg.norm(err) < pos_tol:
                break
            mujoco.mj_jacSite(m, d, jp, None, site)
            J = jp[:, dof]
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), err)
            d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, lo, hi)
        resid = float(np.linalg.norm(target_pos - d.site_xpos[site]))
        return d.qpos[qadr].copy(), resid


# ----------------------------------------------------------------------------- #
#  identity -> card   (stub recogniser)
# ----------------------------------------------------------------------------- #

IDENTITY_TO_CARD = {"EMP-0007": 0, "EMP-0142": 1, "VISITOR": 2, "EMP-0088": 3, "EMP-0231": 4}
GUEST_IDENTITY = "EMP-0142"          # who is standing at the counter this run


def recognise(rgb_frame, frames_seen):
    """Plug a real face recogniser here. The sim guest is featureless, so we just
    return the known identity after ~1 s of the camera being on target."""
    if frames_seen < 60:
        return None
    return GUEST_IDENTITY


# ----------------------------------------------------------------------------- #
#  demo controller (state machine)
# ----------------------------------------------------------------------------- #

class Demo:
    def __init__(self, scene: Scene, cam_renderer=None):
        self.s = scene
        self.cam = cam_renderer
        self.dt = scene.model.opt.timestep
        self.state = "START"
        self.t = 0.0
        self.t0 = 0.0
        self.just_entered = True
        self._last_handled = None
        self.frames_seen = 0
        self.card_idx = None
        self.held = None            # (card_idx, rel_pos, rel_quat)  when a card is grasped

        # per-arm smooth move: ctrl interpolation
        m = scene.model
        self.q_from = {"left": scene.data.ctrl[scene.arm["left"]["act"]].copy(),
                       "right": scene.data.ctrl[scene.arm["right"]["act"]].copy()}
        self.q_to = {"left": self._arm_ctrl("left"), "right": self._arm_ctrl("right")}
        self.mv_t0 = {"left": 0.0, "right": 0.0}
        self.mv_dur = {"left": 1.0, "right": 1.0}
        self.grip_cmd = {"left": scene.GRIP_OPEN, "right": scene.GRIP_OPEN}

        # a comfortable "ready" pose for both arms (also the IK seed)
        self.READY = Scene.IK_SEED.copy()
        # left-arm "scanning" pose: puts left_wrist_cam ~0.64 m from the guest's
        # face, pointing straight at it (found by search, aim error < 1 deg).
        self.SCAN_POSE = np.array([-0.40, 0.94, -0.48, 0.0, -0.80, 1.50])

    # ---- helpers ---------------------------------------------------------- #
    def reset(self):
        """Restart the scenario: cards back in the rack, arms back to zero."""
        s = self.s
        mujoco.mj_resetData(s.model, s.data)
        for _ in range(1200):                        # let the card fan settle
            mujoco.mj_step(s.model, s.data)
        self.state, self._last_handled = "START", None
        self.just_entered, self.frames_seen = True, 0
        self.card_idx, self.held = None, None
        for arm in ("left", "right"):
            self.q_from[arm] = np.zeros(6)
            self.q_to[arm] = np.zeros(6)
            self.mv_t0[arm], self.mv_dur[arm] = self.t, 1.0
            self.grip_cmd[arm] = s.GRIP_OPEN
        print(f"[{self.t:6.2f}s]  === restarting scenario ===")

    def _arm_ctrl(self, arm):
        return self.s.data.ctrl[self.s.arm[arm]["act"]].copy()

    def move_arm(self, arm, q_target, dur=1.5):
        self.q_from[arm] = self._arm_ctrl(arm)
        self.q_to[arm] = np.asarray(q_target, float)
        self.mv_t0[arm] = self.t
        self.mv_dur[arm] = dur

    def arm_settled(self, arm, tol=0.03):
        q = self.s.data.qpos[self.s.arm[arm]["q"]]
        return np.max(np.abs(q - self.q_to[arm])) < tol and \
            (self.t - self.mv_t0[arm]) > self.mv_dur[arm]

    def ik_move(self, arm, pos, dur=1.5, seed_arm=None):
        q, resid = self.s.solve_ik(arm, pos, seed_arm=seed_arm)
        self.move_arm(arm, q, dur)
        return resid

    # ---- main tick ------------------------------------------------------- #
    def step(self):
        s = self.s
        m, d = s.model, s.data

        # 1) interpolate each arm's ctrl toward its target, set gripper
        for arm in ("left", "right"):
            a = s.arm[arm]
            alpha = np.clip((self.t - self.mv_t0[arm]) / max(self.mv_dur[arm], 1e-6), 0, 1)
            alpha = alpha * alpha * (3 - 2 * alpha)          # smoothstep
            d.ctrl[a["act"]] = (1 - alpha) * self.q_from[arm] + alpha * self.q_to[arm]
            d.ctrl[a["grip"]] = self.grip_cmd[arm]

        # 2) advance the state machine  (just_entered = first tick in this state)
        handled = self.state
        self.just_entered = (handled != self._last_handled)
        getattr(self, "st_" + handled.lower())()
        self._last_handled = handled

        # 3) physics
        mujoco.mj_step(m, d)

        # 4) keep a held card attached -- to the gripper ("gripper") while carried,
        #    or frozen in the guest's hand ("hand") after the hand-off.
        if self.held is not None:
            idx = self.held["idx"]
            if self.held["mode"] == "gripper":
                tcp_p, tcp_R = s.tcp_pose("right")
                tcp_q = np.zeros(4); mujoco.mju_mat2Quat(tcp_q, tcp_R.flatten())
                wp = tcp_p + tcp_R @ self.held["rel_pos"]
                wq = np.zeros(4); mujoco.mju_mulQuat(wq, tcp_q, self.held["rel_quat"])
            else:                                       # "hand" -- fixed world pose
                wp, wq = self.held["pos"], self.held["quat"]
            qa = s.card_qadr[idx]
            d.qpos[qa:qa + 3] = wp
            d.qpos[qa + 3:qa + 7] = wq
            dofadr = m.jnt_dofadr[m.body_jntadr[s.card_bid[idx]]]
            d.qvel[dofadr:dofadr + 6] = 0.0
            mujoco.mj_forward(m, d)

        self.t += self.dt

    # ---- grasp / hand-off ------------------------------------------------- #
    @staticmethod
    def _mulq(a, b):
        out = np.zeros(4); mujoco.mju_mulQuat(out, a, b); return out

    def grab(self, idx):
        s = self.s
        tcp_p, tcp_R = s.tcp_pose("right")
        b = s.card_bid[idx]
        cp = s.data.xpos[b].copy()
        cq = np.zeros(4); mujoco.mju_mat2Quat(cq, s.data.xmat[b].copy())
        tcp_q = np.zeros(4); mujoco.mju_mat2Quat(tcp_q, tcp_R.flatten())
        inv_tcp_q = np.zeros(4); mujoco.mju_negQuat(inv_tcp_q, tcp_q)
        self.held = dict(idx=idx, mode="gripper",
                         rel_pos=tcp_R.T @ (cp - tcp_p),
                         rel_quat=self._mulq(inv_tcp_q, cq))

    def handoff(self):
        """Transfer the card from the gripper into the guest's hand (freeze it there)."""
        s = self.s
        b = s.card_bid[self.held["idx"]]
        cq = np.zeros(4); mujoco.mju_mat2Quat(cq, s.data.xmat[b].copy())
        self.held = dict(idx=self.held["idx"], mode="hand",
                         pos=s.data.xpos[b].copy(), quat=cq)

    def release(self):
        self.held = None

    # ---- states ---------------------------------------------------------- #
    def goto(self, state):
        self.state = state
        self.t0 = self.t
        print(f"[{self.t:6.2f}s]  -> {state}")

    def st_start(self):
        if self.just_entered:
            self.move_arm("left", self.READY, 2.0)
            self.move_arm("right", self.READY, 2.0)
        if self.arm_settled("left", tol=0.05) and self.arm_settled("right", tol=0.05):
            self.goto("SCAN")

    def st_scan(self):
        if self.just_entered:
            self.move_arm("left", self.SCAN_POSE, dur=2.5)   # point wrist cam at guest
        if self.arm_settled("left", tol=0.08):
            frame = None
            if self.cam is not None:
                self.cam.update_scene(self.s.data, camera="left_wrist_cam")
                frame = self.cam.render()
            self.frames_seen += 1
            ident = recognise(frame, self.frames_seen)
            if ident is not None:
                self.card_idx = IDENTITY_TO_CARD[ident]
                print(f"          RECOGNISED {ident}  ->  staff_card_{self.card_idx+1}")
                self.goto("APPROACH")

    def st_approach(self):
        if self.just_entered:
            top = self.s.card_top(self.card_idx)
            self.pre = top + np.array([0.0, 0.0, 0.06])
            r = self.ik_move("right", self.pre, dur=2.0)
            print(f"          right pre-grasp IK residual = {r*1000:.1f} mm")
            self.grip_cmd["right"] = self.s.GRIP_OPEN
        if self.arm_settled("right", tol=0.05):
            self.goto("DESCEND")

    def st_descend(self):
        if self.just_entered:
            top = self.s.card_top(self.card_idx)
            self.grasp_pt = top + np.array([0.0, 0.0, -0.004])
            r = self.ik_move("right", self.grasp_pt, dur=1.5,
                             seed_arm=self.s.data.qpos[self.s.arm["right"]["q"]])
            print(f"          right descend IK residual = {r*1000:.1f} mm")
        if self.t - self.t0 > 1.7:
            self.goto("GRASP")

    def st_grasp(self):
        if self.just_entered:
            self.grip_cmd["right"] = self.s.GRIP_SHUT
        if self.t - self.t0 > 0.6 and self.held is None:
            self.grab(self.card_idx)
            print(f"          grasped staff_card_{self.card_idx+1}")
        if self.t - self.t0 > 0.9:
            self.goto("LIFT")

    def st_lift(self):
        if self.just_entered:
            up = self.grasp_pt + np.array([0.0, 0.0, 0.14])
            self.ik_move("right", up, dur=1.5,
                         seed_arm=self.s.data.qpos[self.s.arm["right"]["q"]])
        if self.t - self.t0 > 1.7:
            self.goto("PRESENT")

    def st_present(self):
        if self.just_entered:
            offer = np.array([0.40, -0.12, 1.00])            # raised, out over the counter
            r = self.ik_move("right", offer, dur=2.2)
            print(f"          right present IK residual = {r*1000:.1f} mm")
            self.move_arm("left", self.READY, 2.0)           # left arm done -> relax
        if self.arm_settled("right", tol=0.06):
            self.goto("REACHOUT")

    def st_reachout(self):
        # extend the last few cm so the card sits in the guest's outstretched hand
        if self.just_entered:
            target = self.s.guest_hand() + np.array([0.0, 0.0, 0.055])
            r = self.ik_move("right", target, dur=1.6,
                             seed_arm=self.s.data.qpos[self.s.arm["right"]["q"]])
            print(f"          right reach-to-hand IK residual = {r*1000:.1f} mm")
        if self.arm_settled("right", tol=0.05) or self.t - self.t0 > 2.0:
            self.goto("HANDOFF")

    def st_handoff(self):
        if self.just_entered:
            self.grip_cmd["right"] = self.s.GRIP_OPEN         # let go
        if self.t - self.t0 > 0.45 and self.held is not None \
                and self.held["mode"] == "gripper":
            self.handoff()                                    # card stays in the guest's hand
            print(f"          handed staff_card_{self.card_idx+1} to the guest")
        if self.t - self.t0 > 1.3:
            self.goto("RETRACT")

    def st_retract(self):
        if self.just_entered:
            self.move_arm("right", self.READY, 2.0)
        if self.arm_settled("right", tol=0.05):
            self.goto("DONE")

    def st_done(self):
        pass


# ----------------------------------------------------------------------------- #
#  entry point
# ----------------------------------------------------------------------------- #

class FrameSink:
    """Write frames to out/<name>.mp4 via imageio if available, else a PNG sequence."""

    def __init__(self, name, fps=30):
        self.dir = os.path.join(HERE, "out")
        os.makedirs(self.dir, exist_ok=True)
        self.name, self.fps, self.i = name, fps, 0
        self.writer = None
        try:
            import imageio
            self.writer = imageio.get_writer(
                os.path.join(self.dir, f"{name}.mp4"), fps=fps)
        except Exception:
            self.png_dir = os.path.join(self.dir, name)
            os.makedirs(self.png_dir, exist_ok=True)

    def add(self, frame):
        if self.writer is not None:
            self.writer.append_data(frame)
        else:
            from PIL import Image
            Image.fromarray(frame).save(os.path.join(self.png_dir, f"{self.i:05d}.png"))
        self.i += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print(f"wrote out/{self.name}.mp4  ({self.i} frames)")
        else:
            print(f"wrote out/{self.name}/*.png  ({self.i} frames) "
                  f"-- `pip install imageio[ffmpeg]` for a direct .mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true",
                    help="record third-person + wrist-cam to out/")
    ap.add_argument("--headless", action="store_true", help="no interactive viewer")
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()

    scene = Scene()
    m, d = scene.model, scene.data

    wrist_r = mujoco.Renderer(m, 480, 640)
    demo = Demo(scene, cam_renderer=wrist_r)

    sinks = None
    third_r = None
    if args.video:
        third_r = mujoco.Renderer(m, 720, 1280)
        sinks = dict(third=FrameSink("third_person", 30),
                     wrist=FrameSink("left_wrist_cam", 30))

    def record(k):
        if sinks and k % 8 == 0:                     # 500 Hz sim -> ~60 fps, sample to 30
            third_r.update_scene(d, camera="demo_view")
            sinks["third"].add(third_r.render())
            wrist_r.update_scene(d, camera="left_wrist_cam")
            sinks["wrist"].add(wrist_r.render())

    n_steps = int(args.seconds / m.opt.timestep)
    done_at = None

    if args.headless:
        for k in range(n_steps):
            demo.step()
            record(k)
            if demo.state == "DONE" and done_at is None:
                done_at = demo.t
            if done_at is not None and demo.t > done_at + 2.0:
                break
        if sinks:
            sinks["third"].close()
            sinks["wrist"].close()
        return

    try:
        viewer_cm = mujoco.viewer.launch_passive(m, d)
    except RuntimeError as e:
        print("\n" + "=" * 68)
        print("Interactive viewer needs mjpython on macOS. Run:")
        print("    mjpython issue_card_demo.py")
        print("or record video instead:")
        print("    python3 issue_card_demo.py --headless --video")
        print("=" * 68)
        raise SystemExit(1) from e

    with viewer_cm as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 150, -20, 2.2
        v.cam.lookat[:] = [0.35, 0.0, 0.9]
        k = 0
        while v.is_running() and k < n_steps:
            demo.step()
            if k % 2 == 0:
                v.sync()
            record(k)
            k += 1
            if demo.state == "DONE":                # loop the scenario
                if done_at is None:
                    done_at = demo.t
                if demo.t > done_at + 3.0:
                    demo.reset()
                    done_at = None

    if sinks:
        sinks["third"].close()
        sinks["wrist"].close()


if __name__ == "__main__":
    main()
