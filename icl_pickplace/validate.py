"""用官方 LeRobot v3 reader 验证生成的数据集 + ICL 元数据一致性。

用法: python3 -m icl_pickplace.validate --dataset demos/icl_smoke
"""

import argparse
import json
from pathlib import Path

import numpy as np


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail=""):
        self.items.append((name, bool(ok), detail))

    def summary(self):
        passed = sum(ok for _, ok, _ in self.items)
        print("\n===== 验证结果 =====")
        for name, ok, detail in self.items:
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {name}" + (f"  ({detail})" if detail else ""))
        print(f"共 {len(self.items)} 项, 通过 {passed} 项")
        return passed == len(self.items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="数据集根目录")
    parser.add_argument("--decode-frames", type=int, default=4, help="每集解码验证的帧数")
    args = parser.parse_args()

    root = Path(args.dataset)
    checks = Checks()

    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    repo_id = json.loads((root / "meta" / "icl" / "summary.json").read_text())["repo_id"]

    # 1. 官方 reader 加载
    ds = LeRobotDataset(repo_id, root=root)
    checks.add("官方 LeRobotDataset 加载", True, f"version={CODEBASE_VERSION}")

    num_eps = len(ds.meta.episodes)
    checks.add("episode 索引有效", num_eps > 0, f"episodes={num_eps}")

    feats = ds.features
    img_shape = tuple(feats["observation.images.agentview"]["shape"])
    checks.add(
        "视频特征 observation.images.agentview",
        feats["observation.images.agentview"]["dtype"] == "video" and len(img_shape) == 3,
        str(img_shape),
    )
    action_shape = tuple(feats["action"]["shape"])
    state_shape = tuple(feats["observation.state"]["shape"])
    checks.add(
        "action 特征",
        feats["action"]["dtype"] == "float32" and len(action_shape) == 1,
        f"shape={action_shape}",
    )
    checks.add(
        "observation.state 特征",
        feats["observation.state"]["dtype"] == "float32" and len(state_shape) == 1,
        f"shape={state_shape}",
    )

    # 2. 每集: 长度/时间戳/视频帧解码/action+state
    all_ok_ts, all_ok_video, all_ok_data = True, True, True
    fps = ds.fps
    frame_counts = {}
    eps_table = ds.meta.episodes
    for ep in range(num_eps):
        i0 = int(eps_table["dataset_from_index"][ep])
        i1 = int(eps_table["dataset_to_index"][ep])
        n = i1 - i0
        frame_counts[ep] = n
        if n <= 0:
            all_ok_data = False
            continue
        ts = np.array([float(ds[i]["timestamp"]) for i in range(i0, i1)])
        if not np.allclose(np.diff(ts), 1.0 / fps, atol=1e-3):
            all_ok_ts = False
        probe = sorted(set([0, n // 2, n - 1]))[: max(1, args.decode_frames)]
        for k in probe:
            s = ds[i0 + k]
            img = s["observation.images.agentview"]
            if tuple(img.shape) != img_shape or img.numel() == 0 or float(img.mean()) < 1e-6:
                all_ok_video = False
            if tuple(s["action"].shape) != action_shape or tuple(s["observation.state"].shape) != state_shape:
                all_ok_data = False
    checks.add("所有 episode 时间戳 1/fps 递增", all_ok_ts)
    checks.add("视频帧可解码且分辨率正确", all_ok_video)
    checks.add("action/state 维度一致", all_ok_data)

    # 3. pair 索引与顺序一致性
    pairs_path = root / "meta" / "icl" / "pairs.jsonl"
    pairs = [json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()]
    checks.add("pairs.jsonl 存在且非空", len(pairs) > 0, f"pairs={len(pairs)}")

    link_ok = all(
        0 <= p["context_episode_index"] < num_eps and 0 <= p["query_episode_index"] < num_eps
        for p in pairs
    )
    checks.add("Context/Query 指向有效 episode 索引", link_ok)
    checks.add("每个 pair 记录 intended_object_order", all(p.get("intended_object_order") for p in pairs))

    order_ok, success_ok, evt_ok, frames_ok = True, True, True, True
    for p in pairs:
        for role in ("context", "query"):
            ep = p[f"{role}_episode_index"]
            meta = json.loads(
                (root / "meta" / "icl" / "episodes" / f"episode_{ep:06d}.json").read_text()
            )
            if meta["intended_order"] != p["intended_object_order"]:
                order_ok = False
            if p[f"{role}_success"] != meta["episode_success"]:
                success_ok = False
            if meta["episode_success"]:
                if meta["actual_place_order"] != meta["intended_order"]:
                    success_ok = False
                if meta["pair_id"] != p["pair_id"] or meta["role"] != role:
                    success_ok = False
            place_events = [e["object"] for e in meta["events"] if e["type"] == "place_success"]
            if place_events != meta["actual_place_order"]:
                evt_ok = False
            if meta["episode_success"] and place_events != meta["intended_order"]:
                evt_ok = False
            if meta["num_frames"] != frame_counts.get(ep):
                frames_ok = False
    checks.add("episode 元数据 intended order 与 pair 一致", order_ok)
    checks.add("成功标签 = 实际顺序 = intended (失败不静默标成功)", success_ok)
    checks.add("place_success 事件序列与实际放置顺序一致", evt_ok)
    checks.add("元数据 num_frames 与 parquet 帧数一致", frames_ok)

    ok = checks.summary()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
