"""生成 Context-Query 成对数据集 (LeRobot v3, 支持多进程并行 + 官方聚合)。

用法:
    python3 -m icl_pickplace.generate --num-pairs 50 --num-workers 8 \
        --out-dir demos/icl_data --seed-base 1000

每个 pair: 采样一组 3-5 个随机物体(共享给 context/query) + 独立布局。
--num-workers > 1 时各 worker 生成独立分片数据集, 最后用官方
aggregate_datasets 合并为单一 LeRobot v3 数据集 (meta/icl/ 一并重建)。
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-pairs", type=int, default=2)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--repo-id", type=str, default="icl/pickplace_paired")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--keep-failed", action="store_true")
    p.add_argument("--num-objects-min", type=int, default=3)
    p.add_argument("--num-objects-max", type=int, default=5)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--crf", type=int, default=18, help="h264 质量, 越小越清晰 (LeRobot 默认 30)")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--robot", type=str, default="panda", choices=["panda", "xarm6_robotiq"])
    p.add_argument("--keep-shards", action="store_true", help="保留 worker 分片目录")
    return p.parse_args()


def pair_config(args, pair_id):
    """按 pair_id 确定性采样: 物体数量 + 物体池(共享给 context/query) + 布局种子。"""
    from .env import sample_object_set
    rng = np.random.default_rng(args.seed_base + pair_id * 7919)
    n = int(rng.integers(args.num_objects_min, args.num_objects_max + 1))
    allowed = ["box"] if args.robot == "xarm6_robotiq" else None  # xarm6 圆桶类仍不稳定, 暂只 box
    object_set = sample_object_set(rng, n, allowed_receptacles=allowed)
    intended_order = [s["key"] for s in rng.permutation(object_set["objects"])]
    context_seed = int(rng.integers(1, 2**31 - 1))
    query_seed = int(rng.integers(1, 2**31 - 1))
    return object_set, intended_order, context_seed, query_seed


def run_worker(args, worker_id: int, pair_ids: list, shard_dir: Path) -> dict:
    """单个 worker: 独立 env + 独立 LeRobot 分片数据集。"""
    from .env import ICLPickPlaceEnv
    from .expert import ICLSolver, run_episode
    from .lerobot_export import ICLLeRobotWriter

    env = ICLPickPlaceEnv(robot=args.robot, width=args.width, height=args.height)
    solver = ICLSolver(env)
    nq = env.agent.robot.get_qpos().shape[-1]
    writer = ICLLeRobotWriter(
        root=shard_dir, repo_id=f"{args.repo_id}_shard{worker_id}",
        fps=20, width=args.width, height=args.height, crf=args.crf,
        state_dim=2 * nq + 7, action_dim=len(env.agent.arm_joint_names) + 1,
        robot=args.robot,
    )
    stats = dict(worker=worker_id, attempted=0, valid=0, context_ok=0, query_ok=0,
                 grasp_ok=0, grasp_total=0, place_ok=0, place_total=0,
                 failures={}, timings=[], frames=0)
    for pair_id in pair_ids:
        t0 = time.time()
        object_set, intended_order, c_seed, q_seed = pair_config(args, pair_id)
        stats["attempted"] += 1
        results = {}
        for role, base_seed in (("context", c_seed), ("query", q_seed)):
            ok, rec, used = False, None, base_seed
            for attempt in range(args.max_retries):
                used = base_seed + attempt * 100003
                rec = run_episode(env, solver, intended_order, used, object_set=object_set)
                stats["frames"] += len(rec.frames)
                for g in rec.grasp_results:
                    if not g.get("retry", False):
                        stats["grasp_total"] += 1
                        stats["grasp_ok"] += int(g["success"])
                stats["place_total"] += len(
                    [r for r in rec.place_results if r.get("attempt", 1) == 1])
                stats["place_ok"] += sum(
                    int(r["success"]) for r in rec.place_results if r.get("attempt", 1) == 1)
                if rec.episode_success:
                    ok = True
                    break
                key = rec.failure_reasons[0].split(":")[0] if rec.failure_reasons else "unknown"
                stats["failures"][key] = stats["failures"].get(key, 0) + 1
            results[role] = (ok, rec, used)
        c_ok, c_rec, c_used = results["context"]
        q_ok, q_rec, q_used = results["query"]
        if c_ok:
            stats["context_ok"] += 1
        if q_ok:
            stats["query_ok"] += 1
        if (c_ok and q_ok) or args.keep_failed:
            ci = writer.add_episode(c_rec, pair_id, "context", c_used)
            qi = writer.add_episode(q_rec, pair_id, "query", q_used)
            writer.add_pair(pair_id, ci, qi, intended_order, c_used, q_used,
                            c_ok, q_ok, "valid" if (c_ok and q_ok) else "failed")
            if c_ok and q_ok:
                stats["valid"] += 1
        stats["timings"].append(time.time() - t0)
    writer.finalize()
    stats["episodes"] = 2 * stats["valid"] + (
        2 * (stats["attempted"] - stats["valid"]) if args.keep_failed else 0)
    return stats


def merge_shards(out_dir: Path, args, shard_dirs, shard_eps) -> None:
    """官方 aggregate 合并 + 重建 meta/icl 辅助元数据 (episode_index 偏移重编号)。"""
    from lerobot.datasets.aggregate import aggregate_datasets

    # 跳过空分片 (0 有效 pair 的 worker 不会产生合法数据集)
    pairs_meta = [(i, d) for i, d in enumerate(shard_dirs) if shard_eps[d] > 0]
    repo_ids = [f"{args.repo_id}_shard{i}" for i, _ in pairs_meta]
    roots = [d for _, d in pairs_meta]
    aggregate_datasets(repo_ids=repo_ids, aggr_repo_id=args.repo_id,
                       roots=roots, aggr_root=out_dir)
    # 重建 meta/icl
    icl_out = out_dir / "meta" / "icl"
    (icl_out / "episodes").mkdir(parents=True, exist_ok=True)
    all_pairs, offset = [], 0
    for shard in shard_dirs:
        s_icl = shard / "meta" / "icl"
        n_eps = shard_eps[shard]
        for i in range(n_eps):
            src = s_icl / "episodes" / f"episode_{i:06d}.json"
            meta = json.loads(src.read_text())
            meta["episode_index"] = offset + i
            (icl_out / "episodes" / f"episode_{offset + i:06d}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1))
        if (s_icl / "pairs.jsonl").exists():
            for line in (s_icl / "pairs.jsonl").read_text().splitlines():
                if line.strip():
                    pair = json.loads(line)
                    pair["context_episode_index"] += offset
                    pair["query_episode_index"] += offset
                    all_pairs.append(pair)
        offset += n_eps
    with open(icl_out / "pairs.jsonl", "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    # 汇总 summary (复用第一个分片的 schema 说明)
    from .lerobot_export import SCHEMA_MD
    (icl_out / "SCHEMA.md").write_text(SCHEMA_MD.strip() + "\n")
    (icl_out / "summary.json").write_text(json.dumps(
        dict(codebase_version="v3.0", repo_id=args.repo_id, fps=20,
             num_episodes=offset, num_pairs=len(all_pairs),
             resolution=[args.width, args.height], crf=args.crf),
        ensure_ascii=False, indent=1))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"输出目录已存在: {out_dir}")

    all_ids = list(range(args.num_pairs))
    t_start = time.time()

    if args.num_workers <= 1:
        print("[generate] 单进程生成 ...")
        stats = run_worker(args, 0, all_ids, out_dir)
        all_stats = [stats]
        merge_needed = False
    else:
        print(f"[generate] {args.num_workers} workers 并行生成 ...")
        shard_dirs = [out_dir.parent / f"{out_dir.name}_shard{i}"
                      for i in range(args.num_workers)]
        chunks = [list(map(int, c)) for c in np.array_split(all_ids, args.num_workers)] or []
        ctx = mp.get_context("spawn")
        procs, queues = [], []
        for w, ids in enumerate(chunks):
            q = ctx.Queue()
            p = ctx.Process(target=_worker_entry, args=(args, w, list(ids), shard_dirs[w], q))
            p.start(); procs.append(p); queues.append(q)
        all_stats = []
        for p, q in zip(procs, queues):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker 退出码 {p.exitcode}")
            all_stats.append(q.get())
        merge_shards(out_dir, args, shard_dirs,
                     {d: s["episodes"] for d, s in zip(shard_dirs, all_stats)})
        merge_needed = True
        if not args.keep_shards:
            import shutil
            for d in shard_dirs:
                shutil.rmtree(d, ignore_errors=True)

    # 汇总报告
    total = {k: sum(s[k] for s in all_stats)
             for k in ["attempted", "valid", "context_ok", "query_ok",
                       "grasp_ok", "grasp_total", "place_ok", "place_total", "frames"]}
    failures = {}
    for s in all_stats:
        for k, v in s["failures"].items():
            failures[k] = failures.get(k, 0) + v
    timings = [t for s in all_stats for t in s["timings"]]
    wall = time.time() - t_start
    report = {
        **total,
        "failure_reasons": failures,
        "pairs_per_sec": round(total["attempted"] / wall, 3),
        "wall_time_s": round(wall, 1),
        "avg_pair_wall_s": round(float(np.mean(timings)), 2) if timings else 0,
        "context_success_rate": round(total["context_ok"] / max(1, total["attempted"]), 3),
        "query_success_rate": round(total["query_ok"] / max(1, total["attempted"]), 3),
        "grasp_success_rate": round(total["grasp_ok"] / max(1, total["grasp_total"]), 3),
        "place_success_rate": round(total["place_ok"] / max(1, total["place_total"]), 3),
        "resolution": [args.width, args.height],
        "crf": args.crf,
        "num_workers": args.num_workers,
        "fps": 20,
        "dataset_dir": str(out_dir),
        "dataset_size_mb": round(sum(f.stat().st_size for f in out_dir.rglob("*")
                                     if f.is_file()) / 1e6, 2),
    }
    (out_dir / "meta" / "icl").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "icl" / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=lambda o: int(o) if hasattr(o, "item") else str(o)))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n完成: {out_dir}")
    print(f"验证: python3 -m icl_pickplace.validate --dataset {out_dir}")


def _worker_entry(args, w, ids, shard_dir, q):
    try:
        stats = run_worker(args, w, ids, Path(shard_dir))
        q.put(stats)
    except Exception as e:
        import traceback
        traceback.print_exc()
        q.put(dict(worker=w, error=str(e), attempted=0, valid=0, context_ok=0, query_ok=0,
                   grasp_ok=0, grasp_total=0, place_ok=0, place_total=0, frames=0,
                   failures={}, timings=[], episodes=0))
        raise


if __name__ == "__main__":
    main()
