"""离线检查/可视化一个 Context-Query pair (headless)。

用法: python3 -m icl_pickplace.inspect_pair --dataset demos/icl_smoke --pair-id 0
输出: 终端元数据摘要 + side-by-side mp4 (上: Context, 下: Query)
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pair-id", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="输出 mp4 路径")
    args = parser.parse_args()

    root = Path(args.dataset)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = json.loads((root / "meta" / "icl" / "summary.json").read_text())["repo_id"]
    ds = LeRobotDataset(repo_id, root=root)

    pairs = [
        json.loads(l)
        for l in (root / "meta" / "icl" / "pairs.jsonl").read_text().splitlines()
        if l.strip()
    ]
    pair = next((p for p in pairs if p["pair_id"] == args.pair_id), None)
    if pair is None:
        raise SystemExit(f"pair_id={args.pair_id} 不存在; 可用: {[p['pair_id'] for p in pairs]}")

    print(f"pair_id={pair['pair_id']}  intended_order={pair['intended_object_order']}")
    print(
        f"context_episode={pair['context_episode_index']} "
        f"(seed={pair['context_seed']}, success={pair['context_success']})"
    )
    print(
        f"query_episode={pair['query_episode_index']}   "
        f"(seed={pair['query_seed']}, success={pair['query_success']})"
    )

    import imageio

    episode_frames = {}
    for role in ("context", "query"):
        ep = pair[f"{role}_episode_index"]
        meta = json.loads(
            (root / "meta" / "icl" / "episodes" / f"episode_{ep:06d}.json").read_text()
        )
        print(
            f"\n[{role}] ep={ep} frames={meta['num_frames']} "
            f"success={meta['episode_success']} "
            f"grasp={meta['actual_grasp_order']} place={meta['actual_place_order']}"
        )
        if meta["failure_reasons"]:
            print(f"    failures: {meta['failure_reasons']}")
        ev_summary = [f"{e['type']}:{e.get('object', '')}@{e['step']}" for e in meta["events"]]
        print(f"    events: {ev_summary}")

        f0 = int(ds.meta.episodes["dataset_from_index"][ep])
        n = int(ds.meta.episodes["dataset_to_index"][ep]) - f0
        frames = []
        for i in range(n):
            img = ds[f0 + i]["observation.images.agentview"].numpy()
            frames.append(np.transpose(img, (1, 2, 0)).astype(np.uint8))
        episode_frames[role] = frames
        s = ds[f0 + n // 2]
        print(
            f"    中间帧 action[:4]={np.round(s['action'][:4].numpy(), 3).tolist()} "
            f"state[:3]={np.round(s['observation.state'][:3].numpy(), 3).tolist()}"
        )

    # side-by-side (上下拼接, 长度对齐到较短者)
    n = min(len(episode_frames["context"]), len(episode_frames["query"]))
    out = args.out or str(root / f"pair_{args.pair_id}_preview.mp4")
    h, w = episode_frames["context"][0].shape[:2]
    with imageio.get_writer(out, fps=ds.fps, quality=8) as wri:
        for i in range(n):
            canvas = np.zeros((h * 2 + 4, w, 3), np.uint8)
            canvas[:h] = episode_frames["context"][i]
            canvas[h + 4 :] = episode_frames["query"][i]
            wri.append_data(canvas)
    print(f"\n预览视频已保存: {out}")


if __name__ == "__main__":
    main()
