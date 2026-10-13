"""脚本专家策略: mplib 运动规划 + 物理抓放 + 交互事件检测。

事件判定原则 (不靠文件名/人工标注):
- grasp_success: 闭合夹爪并提升后, 物体确实离桌且跟随 TCP 运动
- place_success: 松爪并撤离后, 物体停留在篮内且静止
"""

from dataclasses import dataclass, field
from typing import Optional

import mplib
import numpy as np
import sapien
import trimesh

from mani_skill.envs.sapien_env import BaseEnv

from .env import release_slots

FINGER_DEPTH = 0.025

# 各机器人夹爪动作语义 (pd_joint_pos 下的 gripper 指令值)
GRIPPER_COMMANDS = {
    "panda": dict(open=1.0, close=-1.0),
    "xarm6_robotiq": dict(open=0.0, close=0.81),
}
LIFT_HEIGHT = 0.14
SAFE_TRANSIT_Z = 0.30
RELEASE_ABOVE_RIM_BY_ROBOT = {"panda": 0.01, "xarm6_robotiq": 0.04}  # xarm 手指更长, 释放更高
SETTLE_STEPS = 12
MAX_EPISODE_STEPS = 2800


def _to_np_pose(pose) -> sapien.Pose:
    p, q = pose.p, pose.q
    p = p.cpu().numpy().reshape(-1)[:3] if hasattr(p, "cpu") else np.asarray(p).reshape(-1)[:3]
    q = q.cpu().numpy().reshape(-1)[:4] if hasattr(q, "cpu") else np.asarray(q).reshape(-1)[:4]
    return sapien.Pose(p=p, q=q)


@dataclass
class Frame:
    step: int
    rgb: np.ndarray  # (H, W, 3) uint8
    state: np.ndarray  # qpos(9)+qvel(9)+tcp_p(3)+tcp_q(4) = 25
    action: np.ndarray  # 8: arm qpos cmd(7) + gripper cmd(1)
    object_states: dict  # key -> {"p": [3], "q": [4]}


@dataclass
class EpisodeRecord:
    frames: list = field(default_factory=list)
    events: list = field(default_factory=list)
    intended_order: list = field(default_factory=list)
    actual_grasp_order: list = field(default_factory=list)
    actual_place_order: list = field(default_factory=list)
    grasp_results: list = field(default_factory=list)
    place_results: list = field(default_factory=list)
    episode_success: bool = False
    failure_reasons: list = field(default_factory=list)
    initial_layout: dict = field(default_factory=dict)
    object_set: dict = field(default_factory=dict)


class ICLSolver:
    """封装 mplib 规划与逐步执行, 每步记录观测/动作/物体状态 (支持多机器人)。"""

    def __init__(self, env: BaseEnv):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.agent = self.base_env.agent
        self.robot = self.agent.robot
        self.robot_uid = self.agent.uid
        self.n_arm = len(self.agent.arm_joint_names)
        cmds = GRIPPER_COMMANDS.get(self.robot_uid, dict(open=1.0, close=-1.0))
        self.OPEN, self.CLOSED = cmds["open"], cmds["close"]
        self.control_mode = self.base_env.control_mode
        self.elapsed_steps = 0
        self.gripper_state = self.OPEN
        self.record: Optional[EpisodeRecord] = None
        self.use_point_cloud = True
        self.planner = self._setup_planner()

    def _setup_planner(self) -> mplib.Planner:
        """参考官方 PandaArmMotionPlanningSolver.setup_planner。"""
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [j.get_name() for j in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.agent.urdf_path,
            srdf=self.agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=getattr(self.agent, "ee_link_name", "eef"),
        )
        base = self.robot.pose
        base_p = base.p.cpu().numpy().reshape(-1)[:3]
        base_q = base.q.cpu().numpy().reshape(-1)[:4]
        planner.set_base_pose(np.concatenate([base_p, base_q]).astype(np.float64))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * 0.6
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * 0.6
        return planner

    # ------------------------------------------------------------------ #
    # 碰撞点云: 桌面 + 其余物体 + 篮子(可选)
    # ------------------------------------------------------------------ #
    def update_collisions(self, exclude_object: Optional[str] = None, include_basket: bool = True):
        pts_list = []

        def add_box(center, half, n):
            box = trimesh.creation.box(
                np.asarray(half) * 2,
                transform=sapien.Pose(p=center).to_transformation_matrix(),
            )
            pts, _ = trimesh.sample.sample_surface(box, n)
            pts_list.append(pts)

        add_box([-0.13, 0.0, -0.05], [0.62, 0.62, 0.05], n=220)
        for key in self.base_env.objects:
            if key == exclude_object:
                continue
            p, q = self.base_env.object_pose(key)
            spec = self.base_env.object_specs[key]
            if spec["shape"] == "cube":
                half = [spec["half"]] * 3
            elif spec["shape"] == "cylinder":
                half = [spec["radius"], spec["radius"], spec["half_length"]]
            else:
                half = list(spec["half_sizes"])
            rot = sapien.Pose(p=p, q=q).to_transformation_matrix()
            box = trimesh.creation.box(np.asarray(half) * 2, transform=rot)
            pts, _ = trimesh.sample.sample_surface(box, 120)
            pts_list.append(pts)
        if include_basket:
            c = self.base_env.basket_center_xy()
            rs = self.base_env.receptacle_spec
            total_h = rs["floor_thick"] + rs["wall_height"]
            eh = [rs["interior_half"][0] + rs["wall_thick"],
                  rs["interior_half"][1] + rs["wall_thick"]]
            add_box([float(c[0]), float(c[1]), total_h / 2],
                    [eh[0], eh[1], total_h / 2], n=260)
        pts = np.vstack(pts_list)
        # 过滤距机器人基座过近的点 (panda_link0 与点云碰撞会导致 invalid start state)
        base_xy = self.robot.pose.p.cpu().numpy().reshape(-1)[:2]
        far = np.linalg.norm(pts[:, :2] - base_xy, axis=1) > 0.30
        self.planner.update_point_cloud(pts[far])

    # ------------------------------------------------------------------ #
    # 带记录的 step / 夹爪 / 路径跟随
    # ------------------------------------------------------------------ #
    def _collect_state(self) -> np.ndarray:
        nq = self.robot.get_qpos().shape[-1]
        qpos = self.robot.get_qpos().cpu().numpy().reshape(-1)[:nq]
        qvel = self.robot.get_qvel().cpu().numpy().reshape(-1)[:nq]
        tcp = self.agent.tcp.pose
        tcp_p = tcp.p.cpu().numpy().reshape(-1)[:3]
        tcp_q = tcp.q.cpu().numpy().reshape(-1)[:4]
        return np.concatenate([qpos, qvel, tcp_p, tcp_q]).astype(np.float32)

    def _collect_objects(self) -> dict:
        out = {}
        for key in self.base_env.objects:
            p, q = self.base_env.object_pose(key)
            out[key] = {"p": p.tolist(), "q": q.tolist()}
        return out

    def step(self, action: np.ndarray):
        if self.record is not None:
            self.record.frames.append(
                Frame(
                    step=self.elapsed_steps,
                    rgb=self.base_env.get_rgb(),
                    state=self._collect_state(),
                    action=np.asarray(action, dtype=np.float32).copy(),
                    object_states=self._collect_objects(),
                )
            )
        self.env.step(action)
        self.elapsed_steps += 1

    def follow_path(self, result, refine_steps: int = 0) -> None:
        n = result["position"].shape[0]
        for i in range(n + refine_steps):
            qpos = result["position"][min(i, n - 1)][: self.n_arm]
            self.step(np.hstack([qpos, self.gripper_state]).astype(np.float32))

    def set_gripper(self, state: float, t: int = 8) -> None:
        self.gripper_state = state
        qpos = self.robot.get_qpos().cpu().numpy()[0][: self.n_arm]
        for _ in range(t):
            self.step(np.hstack([qpos, state]).astype(np.float32))

    def log_event(self, etype: str, object_key: Optional[str] = None, info: Optional[dict] = None):
        if self.record is not None:
            ev = dict(step=self.elapsed_steps, time=self.elapsed_steps / 20.0, type=etype)
            if object_key is not None:
                ev["object"] = object_key
            if info:
                ev.update(info)
            self.record.events.append(ev)

    # ------------------------------------------------------------------ #
    # 规划原语
    # ------------------------------------------------------------------ #
    def move_RRT(self, pose, planning_time: float = 5.0, use_point_cloud: bool = True) -> bool:
        pose = _to_np_pose(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=use_point_cloud,
            planning_time=planning_time,
            planner_name="RRTConnect",
            wrt_world=True,
        )
        if result["status"] != "Success":
            import os
            if os.environ.get("ICL_DEBUG"):
                print(f"[planner] RRTConnect failed: {result['status']} target={np.round(pose.p,3)}")
            return False
        self.follow_path(result)
        return True

    def move_up_safe(self, pose, total_height: float) -> bool:
        """垂直提升: screw -> RRT -> 分段 screw 多级回退 (起点可能贴近桌面/点云)。"""
        if self.move_screw(pose, use_point_cloud=False):
            return True
        if self.move_RRT(pose, planning_time=3.0, use_point_cloud=False):
            return True
        # 分段: 基于当前 TCP 高度逐步提到目标
        for h in np.linspace(0.04, total_height, 3):
            cur_z = self.agent.tcp.pose.p.cpu().numpy().reshape(-1)[2]
            staged = sapien.Pose(
                p=[pose.p[0], pose.p[1], min(cur_z + h, pose.p[2])],
                q=pose.q,
            )
            if not self.move_screw(staged, use_point_cloud=False):
                return False
        return True

    def move_screw(self, pose, refine_steps: int = 0, use_point_cloud: bool = True) -> bool:
        pose = _to_np_pose(pose)
        for _ in range(2):
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=use_point_cloud,
            )
            if result["status"] == "Success":
                self.follow_path(result, refine_steps=refine_steps)
                return True
        return False


# ---------------------------------------------------------------------- #
# 单物体: 抓取 -> 搬运 -> 入篮 (真实物理交互 + 事件验证)
# ---------------------------------------------------------------------- #
def _pick_object(solver: ICLSolver, key: str) -> tuple:
    env = solver.base_env
    obj_p, _ = env.object_pose(key)
    if obj_p[2] < 0.003 or abs(obj_p[0]) > 0.45 or abs(obj_p[1]) > 0.6:
        return False, f"object_displaced:{key}"
    approaching = np.array([0, 0, -1])
    # 闭合方向对齐物体朝向 (绕 z 旋转后的局部 y 轴), 保证长方体沿短轴被夹
    obj_p0, obj_q = env.object_pose(key)
    w, x, y, z = obj_q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    closing = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    # 抓取目标 = 物体中心 (官方 PickCube 几何: 手指从两侧包住整个物体,
    # 手指尖触桌时 TCP 停在 ~4.8cm, 属预期行为)
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj_p)

    solver.update_collisions(exclude_object=key, include_basket=True)
    # 先垂直升到安全高度, 再横移, 避免低空扫到其他物体
    cur = solver.agent.tcp.pose
    cur_p = cur.p.cpu().numpy().reshape(-1)[:3]
    if cur_p[2] < SAFE_TRANSIT_Z:
        solver.move_up_safe(
            sapien.Pose(p=[cur_p[0], cur_p[1], SAFE_TRANSIT_Z], q=cur.q.cpu().numpy().reshape(-1)[:4]),
            SAFE_TRANSIT_Z - cur_p[2],
        )
    # 注意: grasp 局部 +z 沿接近方向(向下), 负偏移才是向上抬高
    overhead = _to_np_pose(grasp_pose) * sapien.Pose([0, 0, -(LIFT_HEIGHT + 0.04)])
    ok_overhead = False
    for jitter in [(0, 0), (0.008, -0.006), (-0.008, 0.006)]:
        target = sapien.Pose(
            p=[overhead.p[0] + jitter[0], overhead.p[1] + jitter[1], overhead.p[2]],
            q=overhead.q,
        )
        if solver.move_RRT(target, planning_time=6.0):
            ok_overhead = True
            break
    if not ok_overhead:
        return False, f"rrt_to_overhead_failed:{key}"
    pregrasp = _to_np_pose(grasp_pose) * sapien.Pose([0, 0, -0.06])
    if not solver.move_screw(pregrasp, refine_steps=8, use_point_cloud=False):
        if not solver.move_RRT(pregrasp, planning_time=5.0, use_point_cloud=False):
            return False, f"screw_to_pregrasp_failed:{key}"

    solver.log_event("grasp_attempt", key)
    if not solver.move_screw(grasp_pose, refine_steps=10, use_point_cloud=False):
        if not solver.move_RRT(grasp_pose, planning_time=5.0, use_point_cloud=False):
            return False, f"screw_to_grasp_failed:{key}"
    solver.set_gripper(solver.CLOSED, t=12)

    lift_pose = _to_np_pose(grasp_pose) * sapien.Pose([0, 0, -LIFT_HEIGHT])
    # 手指可能贴近桌面 (与桌面点云接触), 垂直提升用多级回退规划
    if not solver.move_up_safe(lift_pose, LIFT_HEIGHT):
        return False, f"lift_failed:{key}"

    tcp_p = env.agent.tcp.pose.p.cpu().numpy().reshape(-1)[:3]
    obj_p2, _ = env.object_pose(key)
    # 物理判据: 离桌 >4.5cm 且 xy 跟随 TCP — 只有被真实夹持才能同时满足
    lifted = obj_p2[2] > obj_p[2] + 0.045
    follows = np.linalg.norm(obj_p2[:2] - tcp_p[:2]) < 0.075
    import os
    if os.environ.get("ICL_DEBUG"):
        tail = solver.record.frames[-8:] if len(solver.record.frames) >= 8 else solver.record.frames
        print(f"[grasp-verify {key}] lifted={lifted} (dz={obj_p2[2]-obj_p[2]:.3f}) follows={follows} (d={np.linalg.norm(obj_p2[:2]-tcp_p[:2]):.3f}) obj={np.round(obj_p2,3)} tcp={np.round(tcp_p,3)} fingers={np.round(solver.robot.get_qpos().cpu().numpy()[0][7:],4)}")
        for f in tail[::2]:
            print("   frame", f.step, "obj", np.round(f.object_states[key]['p'],3))
    if lifted and follows:
        solver.log_event("grasp_success", key)
        solver.log_event(
            "object_lifted", key, info=dict(height=float(obj_p2[2] - obj_p[2]))
        )
        return True, ""
    return False, f"grasp_verification_failed:{key}"


def _place_object(solver: ICLSolver, key: str, slot_xy) -> tuple:
    env = solver.base_env
    c = env.basket_center_xy()
    target_xy = np.array([c[0] + slot_xy[0], c[1] + slot_xy[1]], dtype=np.float64)
    grasp_q = env.agent.tcp.pose.q.cpu().numpy().reshape(-1)[:4]
    grasp_q = grasp_q / np.linalg.norm(grasp_q)

    # 先垂直升到安全高度再横移到篮口上方
    cur = solver.agent.tcp.pose
    cur_p = cur.p.cpu().numpy().reshape(-1)[:3]
    if cur_p[2] < SAFE_TRANSIT_Z:
        solver.move_up_safe(
            sapien.Pose(p=[cur_p[0], cur_p[1], SAFE_TRANSIT_Z], q=cur.q.cpu().numpy().reshape(-1)[:4]),
            SAFE_TRANSIT_Z - cur_p[2],
        )
    # 搬运到篮口上方: 起点/终点均 >=0.20m 且姿态不变 (纯平移), 优先快速 screw
    solver.update_collisions(exclude_object=None, include_basket=True)
    wall_h = env.receptacle_spec["wall_height"]
    above_rim = sapien.Pose(
        p=[target_xy[0], target_xy[1], wall_h + 0.13], q=grasp_q
    )
    if not solver.move_screw(above_rim, use_point_cloud=False):
        if not solver.move_RRT(above_rim, planning_time=6.0, use_point_cloud=False):
            return False, f"transit_to_basket_failed:{key}"

    # 下探进篮 (垂直路径安全, 移除篮子障碍)
    solver.update_collisions(exclude_object=None, include_basket=False)
    solver.log_event("place_attempt", key)
    release_z = wall_h + RELEASE_ABOVE_RIM_BY_ROBOT.get(solver.robot_uid, 0.02)
    release_pose = sapien.Pose(p=[target_xy[0], target_xy[1], release_z], q=grasp_q)
    # 垂直下探入篮: 目标槽位在内腔中心区域, 墙壁/桌面点云会造成误碰撞, 关闭点云
    if not solver.move_screw(release_pose, refine_steps=6, use_point_cloud=False):
        if not solver.move_RRT(release_pose, planning_time=5.0, use_point_cloud=False):
            return False, f"screw_to_release_failed:{key}"

    pre_p, _ = env.object_pose(key)
    solver.set_gripper(solver.OPEN, t=10)
    retreat = sapien.Pose(p=[target_xy[0], target_xy[1], release_z + 0.14], q=grasp_q)
    if not solver.move_up_safe(retreat, 0.14):
        return False, f"retreat_failed:{key}"
    solver.set_gripper(solver.OPEN, t=SETTLE_STEPS)

    # 验证: 物体在内腔且静止
    post_p, _ = env.object_pose(key)
    tail = solver.record.frames[-(SETTLE_STEPS + 2):]
    still = True
    if len(tail) >= 2:
        arr = np.array([f.object_states[key]["p"] for f in tail])
        still = bool(np.abs(np.diff(arr, axis=0)).max() < 0.003)
    inside = env.object_inside_receptacle(key)
    if inside and still:
        solver.log_event(
            "place_success", key, info=dict(drop=float(pre_p[2] - post_p[2]))
        )
        return True, ""
    reason = "not_inside" if not inside else "not_still"
    return False, f"place_verification_failed:{key}:{reason}"


def run_episode(env, solver: ICLSolver, intended_order: list, layout_seed: int,
                object_set: dict = None) -> EpisodeRecord:
    """执行一整段按 intended_order 的多物体抓放, 返回带事件与状态的记录。"""
    record = EpisodeRecord(intended_order=list(intended_order))
    solver.record = record
    solver.elapsed_steps = 0
    solver.gripper_state = solver.OPEN

    reset_options = dict(layout_seed=layout_seed)
    if object_set is not None:
        reset_options.update(reconfigure=True, object_set=object_set)
    env.reset(options=reset_options)
    # reconfigure 会重建 agent/robot, 必须刷新 solver 内的引用
    solver.base_env = env.unwrapped
    solver.agent = env.unwrapped.agent
    solver.robot = env.unwrapped.agent.robot

    record.object_set = getattr(env.unwrapped, "current_object_set", {})
    record.initial_layout = dict(
        layout_seed=layout_seed,
        basket_center=env.unwrapped.basket_center_xy().tolist(),
        objects={
            k: {
                "p": env.unwrapped.object_pose(k)[0].tolist(),
                "q": env.unwrapped.object_pose(k)[1].tolist(),
            }
            for k in env.unwrapped.objects
        },
    )

    solver.set_gripper(solver.OPEN, t=6)

    for idx, key in enumerate(intended_order):
        if solver.elapsed_steps > MAX_EPISODE_STEPS:
            record.failure_reasons.append("timeout")
            solver.log_event("episode_failure", info=dict(reason="timeout"))
            break

        grasp_ok, grasp_reason = _pick_object(solver, key)
        record.grasp_results.append(dict(object=key, success=grasp_ok, reason=grasp_reason))
        if not grasp_ok:
            # 重试一次: 张爪撤离后重新接近
            solver.set_gripper(solver.OPEN, t=6)
            safe = _to_np_pose(env.unwrapped.agent.tcp.pose)
            solver.move_screw(safe * sapien.Pose([0, 0, 0.10]))
            grasp_ok, grasp_reason = _pick_object(solver, key)
            record.grasp_results.append(
                dict(object=key, success=grasp_ok, reason=grasp_reason, retry=True)
            )
            if not grasp_ok:
                record.failure_reasons.append(grasp_reason)
                solver.log_event("episode_failure", key, info=dict(reason=grasp_reason))
                break
        record.actual_grasp_order.append(key)

        # 阶段监控: 其余物体应仍在初始位置附近 (未被撞飞)
        for other in intended_order[idx + 1 :]:
            op, _ = env.unwrapped.object_pose(other)
            ip = np.asarray(record.initial_layout["objects"][other]["p"])
            if abs(op[2] - ip[2]) > 0.03 or np.linalg.norm(op[:2] - ip[:2]) > 0.08:
                reason = f"object_displaced_mid_episode:{other}"
                record.failure_reasons.append(reason)
                solver.log_event("episode_failure", other, info=dict(reason=reason))
                break
        if record.failure_reasons:
            break

        place_ok, place_reason = False, "not_attempted"
        for place_attempt in range(2):
            place_ok, place_reason = _place_object(solver, key, release_slots(len(intended_order), env.unwrapped.receptacle_spec["interior_half"], env.unwrapped.receptacle_spec.get("type", "box"))[idx])
            record.place_results.append(
                dict(object=key, success=place_ok, reason=place_reason,
                     attempt=place_attempt + 1)
            )
            if place_ok:
                break
            # 物体若落回桌面 (未入篮), 重新抓起再放一次
            op, _ = env.unwrapped.object_pose(key)
            if op[2] < 0.15 and not env.unwrapped.object_inside_receptacle(key):
                regrasp_ok, regrasp_reason = _pick_object(solver, key)
                record.grasp_results.append(
                    dict(object=key, success=regrasp_ok, reason=regrasp_reason,
                         retry=True, phase="regrasp_after_place_fail")
                )
                if not regrasp_ok:
                    place_reason = regrasp_reason
                    break
            else:
                break
        if not place_ok:
            record.failure_reasons.append(place_reason)
            solver.log_event("episode_failure", key, info=dict(reason=place_reason))
            break
        record.actual_place_order.append(key)

    record.episode_success = (
        len(record.actual_place_order) == len(intended_order)
        and record.actual_place_order == list(intended_order)
    )
    if record.episode_success:
        solver.log_event("episode_success")
    elif not record.failure_reasons:
        record.failure_reasons.append("incomplete")

    # 结尾帧: 用最后一个动作补一帧终态观测 (帧约定见 README)
    if record.frames:
        last = record.frames[-1]
        record.frames.append(
            Frame(
                step=solver.elapsed_steps,
                rgb=env.unwrapped.get_rgb(),
                state=solver._collect_state(),
                action=last.action.copy(),
                object_states=solver._collect_objects(),
            )
        )
    return record
