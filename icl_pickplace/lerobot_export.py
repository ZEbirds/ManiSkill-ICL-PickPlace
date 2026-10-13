"""LeRobot v3 数据集导出器 + ICL 辅助元数据。

主数据集完全遵循官方 LeRobotDataset v3.0 结构 (data/*.parquet, videos/*, meta/*)。
ICL 专有信息放在 meta/icl/ 下 (文档化辅助文件, 不影响官方 reader):
- meta/icl/episodes/episode_XXXXXX.json  每集元数据 (顺序/事件/GT/成功标签)
- meta/icl/pairs.jsonl                    Context-Query 配对索引
- meta/icl/SCHEMA.md                      字段说明
"""

import functools
import json
from pathlib import Path


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")

import numpy as np

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

from .env import CAMERA_EYE, CAMERA_FOV, CAMERA_TARGET

# 语言输入恒定 (不承载任何顺序/身份信息, 仅满足 LeRobot task 字段)
TASK_STRING = "pick_and_place_objects_in_demonstrated_order"

STATE_DIM = 25  # panda: qpos(9)+qvel(9)+tcp(7); 其他机器人按关节数变化
ACTION_DIM = 8


class ICLLeRobotWriter:
    def __init__(self, root: str, repo_id: str, fps: int = 20,
                 width: int = 480, height: int = 360, crf: int = 18,
                 state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 robot: str = "panda"):
        self.state_dim, self.action_dim, self.robot = state_dim, action_dim, robot
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = fps
        self.width = width
        self.height = height
        self.pairs: list = []
        self.episode_count = 0
        # 提升画质: LeRobot 默认 crf=30, 这里降低 crf (值越小质量越高)
        import lerobot.datasets.lerobot_dataset as _ld
        if not getattr(_ld, "_icl_crf_patched", False):
            _orig = _ld.encode_video_frames
            _ld.encode_video_frames = functools.partial(_orig, crf=crf)
            _ld._icl_crf_patched = True
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=self.root,
            use_videos=True,
            vcodec="h264",
            features={
                "observation.images.agentview": {
                    "dtype": "video",
                    "shape": (3, height, width),
                    "names": ["channels", "height", "width"],
                },
                "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
                "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
            },
        )

    def add_episode(self, record, pair_id: int, role: str, seed: int) -> int:
        """写入一个 episode, 返回 LeRobot episode_index。"""
        episode_index = self.episode_count
        for f in record.frames:
            rgb_chw = np.transpose(f.rgb, (2, 0, 1))  # HWC -> CHW
            self.dataset.add_frame(
                {
                    "observation.images.agentview": rgb_chw,
                    "observation.state": np.asarray(f.state, dtype=np.float32),
                    "action": np.asarray(f.action, dtype=np.float32),
                    "task": TASK_STRING,
                }
            )
        self.dataset.save_episode()
        self.episode_count += 1

        meta = {
            "episode_index": episode_index,
            "pair_id": pair_id,
            "role": role,  # context | query
            "seed": seed,
            "fps": self.fps,
            "task_string": TASK_STRING,
            "object_keys": list(record.object_set.get("objects", [])) and
                [s["key"] for s in record.object_set["objects"]],
            "object_specs": {
                s["key"]: s for s in record.object_set.get("objects", [])
            },
            "receptacle_spec": record.object_set.get("receptacle", {}),
            "scene_style": {
                "tabletop": record.object_set.get("tabletop"),
                "backdrop": record.object_set.get("backdrop"),
            },
            "intended_order": list(record.intended_order),
            "actual_grasp_order": list(record.actual_grasp_order),
            "actual_place_order": list(record.actual_place_order),
            "grasp_results": record.grasp_results,
            "place_results": record.place_results,
            "events": record.events,
            "episode_success": bool(record.episode_success),
            "failure_reasons": record.failure_reasons,
            "initial_layout": record.initial_layout,
            "num_frames": len(record.frames),
            "camera": {
                "name": "agentview",
                "width": self.width,
                "height": self.height,
                "fov": CAMERA_FOV,
                "eye": list(CAMERA_EYE),
                "target": list(CAMERA_TARGET),
            },
            "robot": self.robot,
            "state_schema": {
                "names": ["qpos", "qvel", "tcp_pos[3]", "tcp_quat[4]"],
                "dim": self.state_dim,
            },
            "action_schema": {
                "names": ["arm_joint_pos_target", "gripper_cmd[1]"],
                "dim": self.action_dim,
                "gripper_cmd": {"panda": {"open": 1.0, "close": -1.0},
                                 "xarm6_robotiq": {"open": 0.0, "close": 0.81}},
            },
            "frame_convention": "frame_i = (obs_i, action_i): action applied at obs_i leading to obs_{i+1}; last frame repeats last action",
        }
        out = self.root / "meta" / "icl" / "episodes"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"episode_{episode_index:06d}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default)
        )
        return episode_index

    def add_pair(
        self,
        pair_id: int,
        context_index: int,
        query_index: int,
        intended_order: list,
        context_seed: int,
        query_seed: int,
        context_success: bool,
        query_success: bool,
        validation_status: str,
    ):
        self.pairs.append(
            {
                "pair_id": pair_id,
                "context_episode_index": context_index,
                "query_episode_index": query_index,
                "intended_object_order": list(intended_order),
                "context_seed": context_seed,
                "query_seed": query_seed,
                "context_success": bool(context_success),
                "query_success": bool(query_success),
                "validation_status": validation_status,
            }
        )

    def finalize(self) -> Path:
        self.dataset.finalize()
        icl_dir = self.root / "meta" / "icl"
        icl_dir.mkdir(parents=True, exist_ok=True)
        with open(icl_dir / "pairs.jsonl", "w") as f:
            for pair in self.pairs:
                f.write(json.dumps(pair, ensure_ascii=False, default=_json_default) + "\n")
        (icl_dir / "SCHEMA.md").write_text(SCHEMA_MD.strip() + "\n")
        (icl_dir / "summary.json").write_text(
            json.dumps(
                {
                    "codebase_version": CODEBASE_VERSION,
                    "repo_id": self.repo_id,
                    "fps": self.fps,
                    "num_episodes": self.episode_count,
                    "num_pairs": len(self.pairs),
                },
                ensure_ascii=False,
                indent=1,
                default=_json_default,
            )
        )
        return self.root


SCHEMA_MD = """# ICL 辅助元数据 Schema (LeRobot v3 兼容)

主数据集是标准 LeRobot v3.0 (`meta/info.json`, `data/`, `videos/`),
可用官方 `LeRobotDataset(repo_id, root=...)` 直接读取。

本目录存放 ICL 实验专有信息, 全部以 `episode_index` / `pair_id` 显式关联,
不依赖任何文件名约定之外的信息:

- `episodes/episode_XXXXXX.json`: 每集元数据
  - `pair_id`, `role` (context/query), `seed`
  - `intended_order` / `actual_grasp_order` / `actual_place_order` (object asset keys)
  - `grasp_results`, `place_results`, `events` (step/time/type/object)
  - `episode_success`, `failure_reasons`
  - `initial_layout` (objects + receptacle 位姿 GT)
  - `camera`, `state_schema`, `action_schema`, `frame_convention`
- `pairs.jsonl`: 每行一个 pair
  - `pair_id`, `context_episode_index`, `query_episode_index`
  - `intended_object_order`, 两个 seed, 成功标签与 `validation_status`
- `summary.json`: 汇总信息

评测条件 (Full/Endpoint-only/No/Mismatched Context) 可由 `pairs.jsonl` 组合得到:
同数据集内不同 pair 的 context 可作为 Mismatched Context (顺序不同)。
"""
