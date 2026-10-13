# ICL Pick-and-Place 配对数据集生成器 (ManiSkill 3 + LeRobot v3)

本目录实现"视觉/动作上下文学习 (ICL)"研究用的 **Context–Query 成对数据生成管线**:

- 场景: Panda 机械臂 + 3 个视觉可区分物体 (红立方 / 蓝圆柱 / 绿长方体) + 开口篮子
- Context episode: 专家按某个物体顺序依次抓取并放入篮子
- Query episode: 同一组物体、独立随机布局, 相同目标顺序的专家演示 (仅离线评估用)
- 输出: **标准 LeRobot v3.0 数据集** + `meta/icl/` 文档化辅助元数据

## 安装假设

已在当前机器验证过的环境:

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.10 | |
| ManiSkill | 3.0.1 | `~/ManiSkill` 源码运行 |
| torch | 2.11.0+cu128 | 需与 NVIDIA 驱动 (570.x / CUDA 12.8) 匹配 |
| numpy | 1.26.4 | mplib 0.1.1 要求 numpy<2 |
| mplib | 0.1.1 | ManiSkill 官方锁定版本 |
| lerobot | 0.4.4 | `CODEBASE_VERSION=v3.0`, Python≥3.10 的最后版本 |
| torchcodec | 0.11.1 | LeRobot v3 官方视频解码后端 |
| ffmpeg | 4.4 (系统) | torchcodec 运行库 |

安装命令 (如换机器):

```bash
pip install --no-deps lerobot==0.4.4 torchcodec==0.11.1 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
pip install 'datasets>=4.0,<5.0' pandas 'huggingface-hub>=0.34,<0.36' 'av>=15,<16' \
    jsonlines einops termcolor deepdiff 'accelerate>=1.10,<2.0' imageio
sudo apt install ffmpeg
```

> 说明: lerobot 0.4.4 声明 `torch<2.11`/`torchvision<0.26`, 本机用 `--no-deps`
> 安装并实测数据集写入/读取均正常 (见"已知限制")。

## 使用

```bash
cd ~/ManiSkill

# 生成 2 对 (smoke test, 约 1-3 分钟)
python3 -m icl_pickplace.generate --num-pairs 2 --out-dir demos/icl_smoke --seed-base 1000

# 验证 (官方 LeRobot v3 reader + 元数据一致性)
python3 -m icl_pickplace.validate --dataset demos/icl_smoke

# 可视化某个 pair (输出 side-by-side mp4, 无 GUI)
python3 -m icl_pickplace.inspect_pair --dataset demos/icl_smoke --pair-id 0
```

生成器 CLI 参数:

| 参数 | 默认 | 说明 |
|---|---|---|
| `--num-pairs` | 2 | 生成 pair 数 |
| `--num-workers` | 1 | 并行进程数 (官方 aggregate 合并分片; 建议 8-16) |
| `--num-objects-min/max` | 3 / 5 | 每对物体数量范围 |
| `--width` / `--height` | 480 / 360 | 视频分辨率 |
| `--crf` | 18 | h264 质量 (越小越清晰, LeRobot 默认 30) |
| `--out-dir` | 必填 | 输出目录 (LeRobot 数据集根) |
| `--repo-id` | `icl/pickplace_paired` | LeRobot repo_id 元数据 |
| `--seed-base` | 1000 | pair 顺序/布局种子基 |
| `--max-retries` | 2 | 每个角色失败重试次数 |
| `--keep-failed` | off | 同时保留失败 pair (默认只保留成功 pair) |

## 数据结构

标准 LeRobot v3.0 (官方 reader 可直接加载):

```
out_dir/
├── data/chunk-000/file-000.parquet      # index/episode_index/frame_index/timestamp/...
├── videos/observation.images.agentview/chunk-000/file-000.mp4
├── meta/info.json / tasks.parquet / episodes/ ...
└── meta/icl/                            # ICL 辅助元数据 (文档化, 非官方字段)
    ├── SCHEMA.md                        # 字段说明
    ├── episodes/episode_XXXXXX.json     # 每集: 顺序/事件/GT/成功标签
    ├── pairs.jsonl                      # pair -> context/query episode 映射
    ├── summary.json
    └── report.json                      # 成功率/耗时/数据量统计
```

features:

| key | dtype | shape | 说明 |
|---|---|---|---|
| `observation.images.agentview` | video (h264) | (3,240,320) | 固定侧前 3/4 视角相机 (fov=60°, 已验证三物体+篮子全可见) |
| `observation.state` | float32 | (25,) | qpos(9)+qvel(9)+tcp_pos(3)+tcp_quat(4) |
| `action` | float32 | (8,) | 7 关节位置目标 + 夹爪指令(+1 开/-1 闭) |

帧约定: `frame_i = (obs_i, action_i)`, action 为在 obs_i 上执行并转移到 obs_{i+1} 的命令;
结尾帧重复最后一个动作 (用于保留终态图像)。`fps=20`。

task 字符串恒为 `pick_and_place_objects_in_demonstrated_order`
(LeRobot schema 必填字段; 语言对模型无信息量, 满足"无语言条件"要求)。

### 物体顺序与事件

- `intended_order`: 期望顺序; `actual_grasp_order`/`actual_place_order`: 事件检测出的真实顺序
- 事件 (含 step/time/object): `grasp_attempt`, `grasp_success`, `object_lifted`,
  `place_attempt`, `place_success`, `episode_success`, `episode_failure`
- grasp_success 判定: 闭合夹爪并提升后物体离桌 >4.5cm、跟随 TCP、有真实位移
- place_success 判定: 松爪撤离后物体位于篮内 AABB 且连续帧静止
- episode_success 当且仅当实际放置顺序 == intended_order; 失败不静默标成功

### 评测条件支持

由 `pairs.jsonl` 直接构造: Full Context / Endpoint-only (首末帧) / No Context /
Mismatched Context (取其他 pair 的 context, 其顺序不同)。

## 已知限制

1. v1 只有 3 种基元物体和 1 种矩形篮 (篮朝向固定); YCB 等资产留待扩展
2. 篮内投放槽按顺序固定, 保证不堆叠 (简化验证; 后续可随机化)
3. mplib 点云避障不含被抓物体本身, 极端布局下可能擦碰 (重试机制兜底)
4. 蓝色圆柱为横躺姿态生成 (SAPIEN 圆柱轴沿局部 x); 顶部抓取夹持其直径方向
4. LeRobot 0.4.4 的依赖声明与 torch 2.11/torchvision 0.26 不匹配 (用 --no-deps 绕过,
   实测 v3 读写正常; 若未来 reader 出问题, 可降级 torch 至 2.10.0+cu128)
5. `opencv-python 5.0` 声明要 numpy>=2, 与 mplib 需要的 numpy 1.26 冲突,
   实测当前组合可用

## 故障排查

## 实测结果 (2026-10-13, 本机, 多样化版)

- 随机化设置: 3-5 物体/pair (形状 cube/box/cylinder + HSV 随机颜色尺寸)、篮子尺寸/颜色随机、桌面与背景板颜色随机
- 16 pair / 8 workers 基准: **95.6s 墙钟 (0.167 pair/s ≈ 600 pair/小时)**, 15/16 valid
- 抓取成功率 97.1%, 放置成功率 98.5% (重试后)
- 默认画质 480x360 @ crf18 (LeRobot 默认 320x240 @ crf30); 可用 --width 640 --height 480 --crf 15 进一步提升
- 磁盘 ~25MB/pair @ 480x360; 1000 pair 约 1-2 小时 (8-16 workers)、~25GB
- 验证: `validate.py` 15/15 项通过 (含官方聚合后的数据集)
- 夹爪多样化 (多机器人) 暂未实现, ManiSkill 自带 xarm6_robotiq/widowx 可作为后续扩展

## 故障排查

| 现象 | 原因/解决 |
|---|---|
| `mplib ... Segmentation fault` | numpy 被升到 2.x; `pip install numpy==1.26.4` |
| `NVIDIA driver too old` | torch 与驱动不匹配; 570 驱动装 cu128 版 torch |
| `Failed to find Vulkan ICD` 警告 | SAPIEN 误报, 可忽略 (渲染实测正常) |
| LeRobot 读视频报 `torchvision.io` 错误 | 装 `torchcodec` + 系统 `ffmpeg` |
| 抓取失败率高 | 查看报告 `failure_reasons`; 可提高 `--max-retries` |
