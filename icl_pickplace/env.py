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

Q_CYL_VERTICAL = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]  # 绕y转90度: 轴 x->z
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


def sample_object_set(rng: np.random.Generator, num_objects: int,
                      allowed_receptacles=None) -> dict:
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
    # 篮子类型: YCB 碗(真实 mesh)只装 <=3 个物体; 浅托盘墙更矮; 锥形桶上宽下窄
    choices = ["box", "round", "tray", "tapered"]
    if num_objects <= 3:
        choices.append("ycb_bowl")
    if allowed_receptacles is not None:
        choices = [c for c in choices if c in allowed_receptacles] or ["box"]
    receptacle_type = str(rng.choice(choices))
    if receptacle_type == "tray":
        interior = (float(rng.uniform(0.11, 0.13)), float(rng.uniform(0.09, 0.105)))
        wall_height = float(rng.uniform(0.045, 0.055))
    elif receptacle_type == "tapered":
        interior = (float(rng.uniform(0.105, 0.125)), float(rng.uniform(0.085, 0.095)))
        wall_height = float(rng.uniform(0.07, 0.09))
    elif receptacle_type == "ycb_bowl":
        interior = (0.065, 0.065)  # 有效内半径
        wall_height = 0.048
        for o in objects:  # 碗小, 物体半径压到 <=0.033
            if o["shape"] == "cube" and o["half"] > 0.033:
                o["half"], o["radius"], o["top_z"] = 0.033, 0.033, 0.033
            elif o["shape"] == "box":
                hx = min(o["half_sizes"][0], 0.032); hy = min(o["half_sizes"][1], 0.026)
                o["half_sizes"] = [hx, hy, o["half_sizes"][2]]
                o["radius"], o["top_z"] = float(np.hypot(hx, hy)), o["half_sizes"][2]
            elif o["shape"] == "cylinder" and o["radius"] > 0.033:
                o["radius"], o["radius3"] = 0.033, 0.033
    else:
        wall_height = float(rng.uniform(0.065, 0.085))
    receptacle = dict(
        type=receptacle_type,
        interior_half=interior,
        wall_thick=float(rng.uniform(0.008, 0.012)),
        wall_height=wall_height,
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
    table_style = str(rng.choice(["patchwork", "wood_planks", "round_table",
                                  "dark_gloss", "white_matte"]))
    tabletop = dict(style=table_style, base_hue=float(base_hue), cells=cells)
    scene_theme = str(rng.choice(["plain", "home", "factory", "warehouse"]))
    return dict(objects=objects, receptacle=receptacle, tabletop=tabletop,
                backdrop=dict(theme=scene_theme,
                              color=[float(rng.uniform(0.4, 0.85)),
                                     float(rng.uniform(0.4, 0.85)),
                                     float(rng.uniform(0.4, 0.85)), 1.0]))


def quat_mul_wxyz(a, b):
    """四元数乘法 (w,x,y,z)。"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


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
    if receptacle_type == "ycb_bowl":
        # 碗小: 对角布局避免叠放 (最多 3)
        diag = interior_half[0] * 0.45
        pts = [(diag, diag), (-diag, -diag), (diag, -diag)]
        return pts[:num_objects]
    if receptacle_type in ("round", "tapered"):
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
        rtype = self.receptacle_spec.get("type", "box")
        if rtype == "round":
            self.receptacle = self._build_round_receptacle(self.receptacle_spec)
        elif rtype == "tapered":
            self.receptacle = self._build_tapered_receptacle(self.receptacle_spec)
        elif rtype == "ycb_bowl":
            self.receptacle = self._build_ycb_bowl(self.receptacle_spec)
        else:  # box / tray (同一构建器, 参数不同)
            self.receptacle = self._build_receptacle(self.receptacle_spec)

        self._build_tabletop_style(obj_set["tabletop"])

        self._build_backdrop_theme(obj_set["backdrop"])

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
        builder.add_cylinder_collision(pose=sapien.Pose(p=[0, 0, ft / 2], q=Q_CYL_VERTICAL),
                                       radius=r + t, half_length=ft / 2)
        builder.add_cylinder_visual(pose=sapien.Pose(p=[0, 0, ft / 2], q=Q_CYL_VERTICAL),
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

    def _build_tapered_receptacle(self, spec: dict) -> Actor:
        """锥形桶: 12 段壁板向外倾斜, 上口大下口小。"""
        r_top = spec["interior_half"][0]
        r_bot = r_top * 0.62
        t = spec["wall_thick"]
        h = spec["wall_height"]
        ft = spec["floor_thick"]
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=spec["color"])
        builder.add_cylinder_collision(pose=sapien.Pose(p=[0, 0, ft / 2], q=Q_CYL_VERTICAL),
                                       radius=r_bot, half_length=ft / 2)
        builder.add_cylinder_visual(pose=sapien.Pose(p=[0, 0, ft / 2], q=Q_CYL_VERTICAL),
                                    radius=r_bot, half_length=ft / 2, material=mat)
        n_seg = 12
        for i in range(n_seg):
            ang = 2 * np.pi * i / n_seg
            rm = (r_top + r_bot) / 2
            slant_len = np.hypot(r_top - r_bot, h)
            tilt = np.arctan2(r_top - r_bot, h)
            center = [rm * np.cos(ang), rm * np.sin(ang), ft + h / 2]
            yaw = ang + np.pi / 2
            # 组合旋转: 先绕 y 倾斜 tilt, 再绕 z 转 yaw (wxyz 四元数乘法)
            qz = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            qy = [np.cos(tilt / 2), 0, np.sin(tilt / 2), 0]
            q = quat_mul_wxyz(qz, qy)
            builder.add_box_collision(pose=sapien.Pose(p=center, q=q),
                                      half_size=[slant_len / 2 * 1.08, t / 2, 0.004])
            builder.add_box_visual(pose=sapien.Pose(p=center, q=q),
                                   half_size=[slant_len / 2 * 1.08, t / 2, 0.004], material=mat)
        return builder.build_kinematic(name="receptacle")

    def _build_ycb_bowl(self, spec: dict) -> Actor:
        """真实 YCB 碗 mesh (024_bowl): 已下载资产, kinematic 固定。"""
        from pathlib import Path
        base = Path.home() / ".maniskill/data/assets/mani_skill2_ycb/models/024_bowl"
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(base / "textured.obj"))
        builder.add_convex_collision_from_file(str(base / "collision.ply"))
        return builder.build_kinematic(name="receptacle")

    def _build_tabletop_style(self, tp: dict):
        """桌面视觉风格 (纯视觉, 物理碰撞沿用原桌)。"""
        style = tp.get("style", "patchwork")
        hue = tp.get("base_hue", 0.08)

        def mat(rgb, rough=0.7, metal=0.0):
            try:
                return sapien.render.RenderMaterial(
                    base_color=[*rgb, 1.0], roughness=rough, metallic=metal)
            except TypeError:
                return sapien.render.RenderMaterial(base_color=[*rgb, 1.0])

        def add_box(name, p, hs, material, collision=False):
            b = self.scene.create_actor_builder()
            if collision:
                b.add_box_collision(pose=sapien.Pose(p=p), half_size=hs)
            b.add_box_visual(pose=sapien.Pose(p=p), half_size=hs, material=material)
            b.build_kinematic(name=name)

        import colorsys as _cs
        if style == "patchwork":
            x0, dx = -0.72, 1.24 / 3
            y0, dy = -0.55, 1.10 / 2
            for iy in range(2):
                for ix in range(3):
                    c = tp["cells"][iy * 3 + ix][:3]
                    add_box(f"tp_{ix}_{iy}", [x0 + dx * (ix + .5), y0 + dy * (iy + .5), -0.0015],
                            [dx / 2 * 0.98, dy / 2 * 0.98, 0.0025], mat(c))
        elif style == "wood_planks":
            n = 9
            y0, dy = -0.55, 1.10 / n
            for iy in range(n):
                h = (hue + np.random.default_rng(iy).uniform(-0.02, 0.02)) % 1.0
                v = 0.45 + 0.4 * (iy % 2)
                rgb = _cs.hsv_to_rgb(h, 0.35, v)
                add_box(f"tp_plank{iy}", [-0.10, y0 + dy * (iy + .5), -0.0015],
                        [0.62, dy / 2 * 0.965, 0.0025], mat(rgb, rough=0.55))
        elif style == "round_table":
            b = self.scene.create_actor_builder()
            b.add_cylinder_visual(pose=sapien.Pose(p=[-0.05, -0.05, -0.0015], q=Q_CYL_VERTICAL),
                                  radius=0.45, half_length=0.0025,
                                  material=mat(_cs.hsv_to_rgb(hue, 0.4, 0.6)))
            b.build_kinematic(name="tp_round")
            b2 = self.scene.create_actor_builder()
            b2.add_cylinder_visual(pose=sapien.Pose(p=[-0.05, -0.05, -0.47], q=Q_CYL_VERTICAL),
                                   radius=0.05, half_length=0.47, material=mat([0.25, 0.22, 0.2]))
            b2.add_cylinder_visual(pose=sapien.Pose(p=[-0.05, -0.05, -0.90], q=Q_CYL_VERTICAL),
                                   radius=0.22, half_length=0.02, material=mat([0.22, 0.2, 0.18]))
            b2.build_kinematic(name="tp_round_leg")
        elif style == "dark_gloss":
            add_box("tp_gloss", [-0.10, 0.0, -0.0015], [0.62, 0.55, 0.0025],
                    mat([0.12, 0.12, 0.14], rough=0.15, metal=0.6))
        else:  # white_matte
            add_box("tp_matte", [-0.10, 0.0, -0.0015], [0.62, 0.55, 0.0025],
                    mat([0.88, 0.88, 0.9], rough=0.9))

    def _build_backdrop_theme(self, bd: dict):
        """背景场景 (纯视觉): plain 三面板 / home 家庭 / factory 工厂 / warehouse 仓库。"""
        theme = bd.get("theme", "plain")

        def mat(rgb, rough=0.8, metal=0.0):
            try:
                return sapien.render.RenderMaterial(
                    base_color=[*rgb, 1.0], roughness=rough, metallic=metal)
            except TypeError:
                return sapien.render.RenderMaterial(base_color=[*rgb, 1.0])

        def add_box(name, p, hs, material):
            b = self.scene.create_actor_builder()
            b.add_box_visual(pose=sapien.Pose(p=p), half_size=hs, material=material)
            b.build_kinematic(name=name)

        if theme == "plain":
            c = bd.get("color", [0.6, 0.6, 0.6, 1.0])[:3]
            for name, p, hs in [
                ("bd_back", [-0.95, 0.0, 0.45], [0.02, 1.3, 0.75]),
                ("bd_left", [-0.2, 1.25, 0.45], [1.3, 0.02, 0.75]),
                ("bd_right", [-0.2, -1.25, 0.45], [1.3, 0.02, 0.75]),
            ]:
                add_box(name, p, hs, mat(c))
        elif theme == "home":
            wall = mat([0.82, 0.78, 0.72])
            add_box("bd_wall", [-1.0, 0.3, 0.45], [0.02, 1.6, 1.4], wall)
            add_box("bd_wall2", [-0.2, 1.35, 0.45], [1.4, 0.02, 1.4], mat([0.76, 0.72, 0.66]))
            # 家具剪影: 柜子 + 两层书架 + 沙发扶手
            add_box("bd_cabinet", [-0.85, 0.9, -0.15], [0.18, 0.5, 0.78], mat([0.55, 0.38, 0.25]))
            add_box("bd_shelf_l", [-0.5, 1.18, -0.1], [0.55, 0.12, 0.05], mat([0.6, 0.45, 0.3]))
            add_box("bd_shelf_l2", [-0.5, 1.18, 0.18], [0.55, 0.12, 0.05], mat([0.6, 0.45, 0.3]))
            add_box("bd_shelf_v", [-0.5, 1.18, -0.55], [0.55, 0.12, 0.4], mat([0.6, 0.45, 0.3]))
            add_box("bd_shelf_top", [-0.5, 1.18, 0.42], [0.57, 0.14, 0.04], mat([0.62, 0.47, 0.32]))
            add_box("bd_sofa", [-0.1, -1.32, -0.35], [0.7, 0.25, 0.6], mat([0.45, 0.5, 0.62]))
            add_box("bd_sofa_back", [-0.1, -1.52, -0.05], [0.7, 0.12, 0.3], mat([0.42, 0.47, 0.6]))
        elif theme == "factory":
            add_box("bd_wall", [-1.0, 0.0, 0.45], [0.02, 1.7, 1.4], mat([0.55, 0.56, 0.58], rough=0.4, metal=0.5))
            for i, z in enumerate([-0.2, 0.25, 0.7]):
                add_box(f"bd_stripe{i}", [-0.98, 0.0, z], [0.012, 1.68, 0.10],
                        mat([0.42, 0.43, 0.46], rough=0.35, metal=0.6))
            # 货架 + 货箱
            for side, y in [("a", 1.15), ("b", -1.15)]:
                add_box(f"bd_post_l{side}", [-0.55, y, -0.1], [0.04, 0.04, 0.85], mat([0.3, 0.48, 0.25], metal=0.4))
                add_box(f"bd_post_r{side}", [0.25, y, -0.1], [0.04, 0.04, 0.85], mat([0.3, 0.48, 0.25], metal=0.4))
                for j, z in enumerate([-0.35, 0.0, 0.35]):
                    add_box(f"bd_beam{side}{j}", [-0.15, y, z], [0.42, 0.05, 0.035],
                            mat([0.85, 0.55, 0.1], metal=0.3))
                add_box(f"bd_crate{side}", [-0.25, y, 0.08], [0.13, 0.13, 0.07], mat([0.6, 0.35, 0.2]))
                add_box(f"bd_crate2{side}", [0.1, y, 0.08], [0.10, 0.11, 0.06], mat([0.5, 0.3, 0.18]))
            add_box("bd_hazard", [-0.98, 0.0, -0.62], [0.012, 0.8, 0.06], mat([0.9, 0.75, 0.05]))
        else:  # warehouse
            add_box("bd_wall", [-1.0, 0.0, 0.6], [0.02, 1.7, 1.6], mat([0.6, 0.6, 0.62]))
            add_box("bd_wall2", [-0.2, 1.4, 0.6], [1.5, 0.02, 1.6], mat([0.55, 0.55, 0.58]))
            for side, y, x0 in [("a", 1.12, -0.5), ("b", -1.12, 0.1)]:
                for k in range(3):
                    add_box(f"bd_rpost{k}{side}", [x0 + k * 0.4, y, -0.05], [0.035, 0.035, 0.9], mat([0.35, 0.42, 0.55], metal=0.5))
                for j, z in enumerate([-0.4, -0.05, 0.3]):
                    add_box(f"bd_rbeam{j}{side}", [x0 + 0.4, y, z], [0.62, 0.05, 0.035], mat([0.4, 0.5, 0.68], metal=0.4))
                add_box(f"bd_box1{side}", [x0, y, 0.07], [0.12, 0.12, 0.08], mat([0.72, 0.55, 0.35]))
                add_box(f"bd_box2{side}", [x0 + 0.42, y, 0.07], [0.10, 0.10, 0.07], mat([0.65, 0.48, 0.3]))
                add_box(f"bd_box3{side}", [x0 + 0.8, y, 0.06], [0.09, 0.11, 0.06], mat([0.7, 0.52, 0.33]))

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
        if self.receptacle_spec.get("type") in ("round", "tapered", "ycb_bowl"):
            xy_ok = np.hypot(p[0] - c[0], p[1] - c[1]) <= ihx - margin
        else:  # box / tray
            xy_ok = abs(p[0] - c[0]) <= ihx - margin and abs(p[1] - c[1]) <= ihy - margin
        z_ok = ft - 0.005 <= p[2] <= h + 0.012
        return bool(xy_ok and z_ok)
