"""多物体 Pick-and-Place 场景 (Panda + 3-5 个随机物体 + 随机开口篮子)。

多样性设计 (每 pair 采样一次, context/query 共享):
- 物体: 数量 3-5, 形状 cube/box/cylinder, 尺寸与颜色(HSV) 随机, 同场景内颜色可区分
- 篮子: 内腔尺寸/墙高/颜色随机
- 桌面顶板与背景板: 颜色随机 (纯视觉)
- 布局: context/query 独立随机 (layout_seed)

场景随 options["object_set"] 重建 (reset options={"reconfigure": True, ...})。
"""
from typing import Optional
import colorsys

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import build_box, build_cube, build_cylinder
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360
CAMERA_EYE = (0.25, 0.40, 0.55)
CAMERA_TARGET = (-0.08, -0.06, 0.0)
CAMERA_FOV = np.pi / 3

OBJECT_SPAWN_XY = ((-0.20, 0.02), (-0.13, 0.18))
BASKET_SPAWN_XY = ((-0.08, 0.02), (-0.34, -0.30))

HUE_NAMES = [
    (15, "red"), (45, "orange"), (70, "yellow"), (160, "green"),
    (200, "cyan"), (260, "blue"), (320, "purple"), (360, "magenta"),
]


def hue_name(h):
    h = h % 360
    for hi, n in HUE_NAMES:
        if h <= hi:
            return n
    return "red"


def sample_object_set(rng: np.random.Generator, num_objects: int) -> dict:
    """采样一个 pair 共享的场景配置: 物体池 + 篮子 + 桌面板 + 背景板。"""
    objects = []
    hues = []
    for i in range(num_objects):
        for _ in range(100):
            h = rng.uniform(0, 360)
            if all(abs(h - h0) > 40 for h0 in hues):
                hues.append(h)
                break
        sat, val = rng.uniform(0.75, 1.0), rng.uniform(0.7, 0.95)
        rgb = colorsys.hsv_to_rgb(h / 360, sat, val)
        shape = rng.choice(["cube", "box", "cylinder"])
        if shape == "cube":
            half = rng.uniform(0.02, 0.03)
            spec = dict(shape="cube", half=float(half), radius=float(half * 1.42), top_z=float(half))
        elif shape == "box":
            hx, hy, hz = rng.uniform(0.03, 0.045), rng.uniform(0.02, 0.03), rng.uniform(0.015, 0.025)
            spec = dict(shape="box", half_sizes=[float(hx), float(hy), float(hz)],
                        radius=float(np.hypot(hx, hy)), top_z=float(hz))
        else:
            r, hl = rng.uniform(0.025, 0.035), rng.uniform(0.02, 0.03)
            # SAPIEN 圆柱轴沿局部 x; 竖立放置 (绕 y 转 90 度), 静置高度 = half_length
            spec = dict(shape="cylinder", radius=float(r), half_length=float(hl),
                        radius3=float(r), top_z=float(hl))
        spec["color"] = [float(c) for c in rgb] + [1.0]
        spec["hue"] = float(h)
        spec["color_name"] = hue_name(h)
        spec["key"] = f"obj{i:02d}_{spec['shape']}_{spec['color_name']}"
        objects.append(spec)

    interior = (float(rng.uniform(0.10, 0.12)), float(rng.uniform(0.085, 0.10)))
    receptacle_type = str(rng.choice(["box", "round"]))
    receptacle = dict(
        type=receptacle_type,
        interior_half=interior,
        wall_thick=float(rng.uniform(0.008, 0.012)),
        wall_height=float(rng.uniform(0.065, 0.085)),
        floor_thick=0.01,
        color=[float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.25, 0.6)),
               float(rng.uniform(0.2, 0.5)), 1.0],
    )
    # 桌面: 3x2 拼接色块 (同一基调色相 + 每块明度/饱和扰动)
    base_hue = rng.uniform(0, 1)
    cells = []
    for _ in range(6):
        h = (base_hue + rng.uniform(-0.06, 0.06)) % 1.0
        rgb = colorsys.hsv_to_rgb(h, rng.uniform(0.25, 0.55), rng.uniform(0.35, 0.85))
        cells.append([float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0])
    tabletop = dict(style="patchwork_3x2", base_hue=float(base_hue), cells=cells)
    backdrop = [float(rng.uniform(0.4, 0.85)), float(rng.uniform(0.4, 0.85)),
                float(rng.uniform(0.4, 0.85)), 1.0]
    return dict(objects=objects, receptacle=receptacle, tabletop=tabletop, backdrop=backdrop)


def _orientation_quat(shape: str, yaw: float) -> list:
    """物体初始朝向四元数 (w,x,y,z)。圆柱轴沿局部 x, 需绕 y 转 90 度竖立。"""
    qz = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    if shape != "cylinder":
        return qz
    qy = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def release_slots(num_objects: int, interior_half, receptacle_type: str = "box") -> list:
    """篮内投放槽 (相对篮心), 最多 5 个, 保证互不堆叠。圆桶用环形分布。"""
    if receptacle_type == "round":
        r = interior_half[0] * 0.38
        slots = [(0.0, 0.0)]
        for i in range(num_objects - 1):
            a = 2 * np.pi * i / max(1, num_objects - 1)
            slots.append((r * np.cos(a), r * np.sin(a)))
        return slots[:num_objects]
    pts = [(-0.55, -0.5), (0.55, -0.5), (0.0, 0.0), (-0.55, 0.5), (0.55, 0.5)]
    return [(interior_half[0] * px, interior_half[1] * py) for px, py in pts[:num_objects]]


class ICLPickPlaceEnv(BaseEnv):
    """随机化多物体 pick-and-place 环境 (脚本专家策略用)。"""

    agent: Panda

    def __init__(self, *args, robot: str = "panda", width=CAMERA_WIDTH,
                 height=CAMERA_HEIGHT, **kwargs):
        # 注意: super().__init__ 内部会触发 _default_sensor_configs, 分辨率需先设置
        self.sensor_width = width
        self.sensor_height = height
        kwargs.setdefault("robot_uids", robot)
        kwargs.setdefault("control_mode", "pd_joint_pos")
        kwargs.setdefault("obs_mode", "none")
        kwargs.setdefault("reward_mode", "none")
        kwargs.setdefault("sim_backend", "cpu")
        kwargs.setdefault("render_mode", "sensors")
        # 注意: 这些属性在 _load_scene 中赋值, 不能在 super() 之后清空
        self.object_specs: dict = getattr(self, "object_specs", {})
        self.receptacle_spec: dict = getattr(self, "receptacle_spec", {})
        self.scene_style: dict = getattr(self, "scene_style", {})
        self.objects: dict = getattr(self, "objects", {})
        self.receptacle: Optional[Actor] = getattr(self, "receptacle", None)
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=CAMERA_EYE, target=CAMERA_TARGET)
        return [CameraConfig("agentview", pose, self.sensor_width, self.sensor_height,
                             CAMERA_FOV, 0.01, 100.0)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        import os
        if os.environ.get("ICL_DEBUG"): print("[dbg] _load_scene called, options keys:", list(options.keys()) if options else None)
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        obj_set = options.get("object_set") or sample_object_set(
            np.random.default_rng(0), 3
        )
        self.current_object_set = obj_set
        self.object_specs = {s["key"]: s for s in obj_set["objects"]}
        self.objects = {}
        for spec in obj_set["objects"]:
            if spec["shape"] == "cube":
                actor = build_cube(self.scene, half_size=spec["half"],
                                   color=spec["color"], name=spec["key"])
            elif spec["shape"] == "cylinder":
                actor = build_cylinder(self.scene, radius=spec["radius"],
                                       half_length=spec["half_length"],
                                       color=spec["color"], name=spec["key"])
            else:
                actor = build_box(self.scene, half_sizes=spec["half_sizes"],
                                  color=spec["color"], name=spec["key"])
            self.objects[spec["key"]] = actor

        self.receptacle_spec = obj_set["receptacle"]
        if self.receptacle_spec.get("type") == "round":
            self.receptacle = self._build_round_receptacle(self.receptacle_spec)
        else:
            self.receptacle = self._build_receptacle(self.receptacle_spec)

        # 桌面顶板 (纯视觉): 3x2 拼接色块
        tp = obj_set["tabletop"]
        if tp.get("style") == "patchwork_3x2":
            x0, dx = -0.72, 1.24 / 3
            y0, dy = -0.55, 1.10 / 2
            for iy in range(2):
                for ix in range(3):
                    b = self.scene.create_actor_builder()
                    b.add_box_visual(
                        pose=sapien.Pose(p=[x0 + dx * (ix + 0.5), y0 + dy * (iy + 0.5), -0.0015]),
                        half_size=[dx / 2 * 0.98, dy / 2 * 0.98, 0.0025],
                        material=sapien.render.RenderMaterial(base_color=tp["cells"][iy * 3 + ix]),
                    )
                    b.build_kinematic(name=f"tabletop_style_{ix}_{iy}")
        else:
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(
                pose=sapien.Pose(p=[-0.10, 0.0, -0.0015]),
                half_size=[0.62, 0.55, 0.0025],
                material=sapien.render.RenderMaterial(base_color=tp),
            )
            builder.build_kinematic(name="tabletop_style")

        # 背景板 (纯视觉, 三面)
        bd = obj_set["backdrop"]
        mat = sapien.render.RenderMaterial(base_color=bd)
        for name, p, hs in [
            ("backdrop_back", [-0.95, 0.0, 0.45], [0.02, 1.3, 0.75]),
            ("backdrop_left", [-0.2, 1.25, 0.45], [1.3, 0.02, 0.75]),
            ("backdrop_right", [-0.2, -1.25, 0.45], [1.3, 0.02, 0.75]),
        ]:
            b2 = self.scene.create_actor_builder()
            b2.add_box_visual(pose=sapien.Pose(p=p), half_size=hs, material=mat)
            b2.build_kinematic(name=name)

    def _build_receptacle(self, spec: dict) -> Actor:
        ihx, ihy = spec["interior_half"]
        t = spec["wall_thick"]
        h = spec["wall_height"]
        ft = spec["floor_thick"]
        builder = self.scene.create_actor_builder()
        mat_wall = sapien.render.RenderMaterial(base_color=spec["color"])
        mat_floor = sapien.render.RenderMaterial(
            base_color=[min(1, c * 1.15) for c in spec["color"][:3]] + [1.0])

        def box(half, p, material):
            builder.add_box_collision(pose=sapien.Pose(p=p), half_size=half)
            builder.add_box_visual(pose=sapien.Pose(p=p), half_size=half, material=material)

        box([ihx + t, ihy + t, ft / 2], [0, 0, ft / 2], mat_floor)
        box([ihx + t, t / 2, ft + h / 2], [0, ihy + t / 2, ft + h / 2], mat_wall)
        box([ihx + t, t / 2, ft + h / 2], [0, -(ihy + t / 2), ft + h / 2], mat_wall)
        box([t / 2, ihy, ft + h / 2], [ihx + t / 2, 0, ft + h / 2], mat_wall)
        box([t / 2, ihy, ft + h / 2], [-(ihx + t / 2), 0, ft + h / 2], mat_wall)
        return builder.build_kinematic(name="receptacle")

    def _build_round_receptacle(self, spec: dict) -> Actor:
        """圆桶: 10 段弧形壁 (小 box 近似) + 圆形底。interior_half[0] 为内半径。"""
        r = spec["interior_half"][0]
        t = spec["wall_thick"]
        h = spec["wall_height"]
        ft = spec["floor_thick"]
        builder = self.scene.create_actor_builder()
        mat_wall = sapien.render.RenderMaterial(base_color=spec["color"])
        mat_floor = sapien.render.RenderMaterial(
            base_color=[min(1, c * 1.15) for c in spec["color"][:3]] + [1.0])
        # 圆形底 (薄圆柱)
        builder.add_cylinder_collision(pose=sapien.Pose(p=[0, 0, ft / 2]),
                                       radius=r + t, half_length=ft / 2)
        builder.add_cylinder_visual(pose=sapien.Pose(p=[0, 0, ft / 2]),
                                    radius=r + t, half_length=ft / 2, material=mat_floor)
        # 10 段壁
        n_seg, rc = 10, r + t / 2
        chord = 2 * (r + t) * np.sin(np.pi / n_seg) * 1.15
        for i in range(n_seg):
            ang = 2 * np.pi * i / n_seg
            center = [rc * np.cos(ang), rc * np.sin(ang), ft + h / 2]
            yaw = ang + np.pi / 2
            q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            builder.add_box_collision(pose=sapien.Pose(p=center, q=q),
                                      half_size=[chord / 2, t / 2, h / 2])
            builder.add_box_visual(pose=sapien.Pose(p=center, q=q),
                                   half_size=[chord / 2, t / 2, h / 2], material=mat_wall)
        return builder.build_kinematic(name="receptacle")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        import os
        if os.environ.get("ICL_DEBUG"): print(f"[dbg] _initialize_episode: objects={len(self.objects)} receptacle={self.receptacle is not None} reconfig={options.get('reconfigure') if options else None}")
        if self.receptacle is None or not self.objects:
            # 防御: 某些内部 reset 路径可能先于场景加载
            self._load_scene(options if isinstance(options, dict) else {})
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            rng = np.random.default_rng(int(options.get("layout_seed", 0)))
            self._apply_layout(self._sample_layout(rng))

    def _sample_layout(self, rng: np.random.Generator) -> dict:
        (ox0, ox1), (oy0, oy1) = OBJECT_SPAWN_XY
        (bx0, bx1), (by0, by1) = BASKET_SPAWN_XY
        basket_center = np.array([rng.uniform(bx0, bx1), rng.uniform(by0, by1)], dtype=np.float32)
        placements = {}
        for clearance in (0.045, 0.03, 0.015):  # 渐进放宽间距, 避免大物体多数量时采样失败
            placements = {}
            ok = True
            for key, spec in self.object_specs.items():
                r = spec["radius"]
                for _ in range(300):
                    xy = np.array([rng.uniform(ox0, ox1), rng.uniform(oy0, oy1)], dtype=np.float32)
                    if any(np.linalg.norm(xy - p["xy"]) < (r + q["radius"] + clearance)
                           for p, q in zip(placements.values(),
                                           [self.object_specs[k] for k in placements])):
                        continue
                    yaw = float(rng.uniform(0, 2 * np.pi))
                    placements[key] = dict(xy=xy, z=float(spec["top_z"]), yaw=yaw)
                    break
                else:
                    ok = False
                    break
            if ok:
                break
        else:
            raise RuntimeError("采样物体初始位置失败 (区域过小)")
        return dict(basket_center=basket_center, placements=placements)

    def _apply_layout(self, layout: dict):
        for key, actor in self.objects.items():
            p = layout["placements"][key]
            q = _orientation_quat(self.object_specs[key]["shape"], p["yaw"])
            actor.set_pose(Pose.create_from_pq(
                torch.tensor([p["xy"][0], p["xy"][1], p["z"]], dtype=torch.float32),
                torch.tensor(q, dtype=torch.float32)))
            body = actor._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if body is not None:
                body.set_linear_velocity(np.zeros(3))
                body.set_angular_velocity(np.zeros(3))
        bx, by = layout["basket_center"]
        self.receptacle.set_pose(Pose.create_from_pq(
            torch.tensor([bx, by, 0.0], dtype=torch.float32),
            torch.tensor([1, 0, 0, 0], dtype=torch.float32)))

    # ---------------- 观测与几何工具 ----------------
    def get_rgb(self) -> np.ndarray:
        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()
        sensor = self.scene.sensors["agentview"]
        rgb = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)["rgb"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.dtype in (np.float32, np.float64):
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
        return rgb

    def object_pose(self, key: str):
        p = self.objects[key].pose.p
        q = self.objects[key].pose.q
        return (p.cpu().numpy().reshape(-1)[:3], q.cpu().numpy().reshape(-1)[:4])

    def basket_center_xy(self) -> np.ndarray:
        return np.asarray(self.receptacle.pose.p.cpu().numpy().reshape(-1)[:2], dtype=np.float32)

    def object_inside_receptacle(self, key: str) -> bool:
        p, _ = self.object_pose(key)
        c = self.basket_center_xy()
        ihx, ihy = self.receptacle_spec["interior_half"]
        ft = self.receptacle_spec["floor_thick"]
        h = self.receptacle_spec["wall_height"]
        margin = 0.004
        if self.receptacle_spec.get("type") == "round":
            xy_ok = np.hypot(p[0] - c[0], p[1] - c[1]) <= ihx - margin
        else:
            xy_ok = abs(p[0] - c[0]) <= ihx - margin and abs(p[1] - c[1]) <= ihy - margin
        z_ok = ft - 0.005 <= p[2] <= h + 0.012
        return bool(xy_ok and z_ok)
