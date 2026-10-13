# ICL Pick-and-Place Paired Data Generator

基于 ManiSkill 3 生成 visual/action in-context learning 用的 **paired data**
（Context–Query 成对 demonstration），输出标准 **LeRobot v3** 数据集。

## 1. 数据内容

| 字段 | 内容 |
|---|---|
| `observation.images.agentview` | 固定外视角 RGB（h264 video 流） |
| `action` | 每步真实下发的关节位置指令 + 夹爪开合指令 |
| `observation.state` | 关节 qpos/qvel + TCP 位姿 |
| `timestamp` / `frame_index` / `episode_index` | 官方 v3 schema，20Hz 严格等间隔 |
| `meta/icl/`（辅助） | 物体 GT 位姿、抓取顺序、交互事件、成功标签、场景配置 |

规格：480×360 @ crf18（可用 `--width/--height/--crf` 调至 640×480 / crf15）；
fps=20；约 25–40 MB/pair。
所有 episodes 的视频连续编码在同一个 mp4 中，读取时按官方
`from_timestamp / to_timestamp` 切分，等效于每集独立视频。

## 2. 数据结构与专家轨迹

目录结构为标准 LeRobot v3（`data/`、`videos/`、`meta/info.json` 等），
官方 reader 可直接加载。辅助元数据约定在 `meta/icl/`（详见其中的 `SCHEMA.md`）：

- `pairs.jsonl`：pair → context/query 两个 episode 的显式映射，
  含 intended 顺序、两个 seed、验证状态
- `episodes/episode_XXXXXX.json`：每集事件日志与 GT
  （抓取/放置事件、真实执行顺序、物体与容器规格）

专家轨迹由脚本策略生成（非遥操作）：

- 长距离移动用 mplib 的 RRT 做避障规划；
  短距离精细接近（如下探抓取）用 screw motion——
  沿螺旋轴的平移+旋转插值，轨迹更直、位姿更准
- 所有 action 在物理仿真中逐步执行；
  抓取验证要求物体真实离桌并跟随 TCP，放置验证要求松爪后静止在容器内
- 只有 context、query 均成功的 pair 才会写入数据集
  （`--keep-failed` 可保留失败样本用于分析）；
  每集的 `success` 标签与失败原因如实记录在元数据中

## 3. 任务设计

当前为最简 **object relocation（pick-and-place）**：

- 桌上 3–5 个物体 + 一个容器；Context 展示抓取顺序，Query 换独立布局按同顺序执行
- 模型只需从 Context 推断抓取顺序：无语言指令、无干扰物体（有几个抓几个）

Diversity（每 pair 随机采样，Context/Query 共享物体集合、独立布局）：

| 维度 | 取值 |
|---|---|
| 物体 | 数量 3–5；cube / box / cylinder；尺寸、HSV 颜色随机（同场景内颜色可区分） |
| 容器 | `box` 开口盒、`round` 圆桶、`tray` 浅托盘、`tapered` 锥形桶、`ycb_bowl` 真实 YCB 碗 mesh |
| 桌面 | 拼色 / 木板条 / 圆桌 / 深色亮面 / 白色哑光 |
| 背景 | 纯色板 / 家庭 / 工厂 / 仓库 |
| 机器人 | panda（全部容器）；xarm6_robotiq（box、round，其余容器适配中） |

## 4. 生成速度与成功率

（32 核 / RTX 4090，CPU 仿真 + 离屏渲染）

- 推荐并行度 **8 workers**（实测 16 workers 无额外收益）
- 吞吐约 **600 pair（1200 episodes）/ 小时**
- 单个 worker 完成一个 pair（context + query，含重试）约 35 s
- 单次 episode 成功率 75–95%（随容器类型、物体数波动），带重试后 pair 有效率通常 ≥ 95%
- 参考批次：16 pair / 96.6 s / 619 MB

## 5. TODO

任务升级：

- [ ] Pose transformation（rotate / reorient / flip；初始姿态、目标角度、旋转轴、物体对称性）
- [ ] Constraint-aware manipulation（insertion、窄空间放置、绕障移动；
      例：方向键插销入槽、bagel 放进 toaster、开下层柜门）
- [ ] 长尾任务（推 toaster 控制杆、刮水器清洁台面）

难度与多样性：

- [ ] 语义级物体（需要类别识别，不只靠颜色/形状）
- [ ] 冗余干扰物体（只抓一部分，考验 Context 指代能力）
- [ ] 相机视角 / 光照随机化；wrist camera；depth & segmentation
- [ ] paired 跨本体（同一顺序 demo 由不同机器人执行）

规模化与工程：

- [ ] 千对级批量采集：分批生成、增量合并、磁盘规划
- [ ] GPU sim 后端（ManiSkill GPU parallel）验证吞吐上限
- [ ] 拟人轨迹：当前 mplib 时间参数化偏机械，改 Hermite / quintic 插值
- [ ] 机器人池扩展（widowx、so100）；xarm6 适配剩余容器
- [ ] train/val/test split 与 eval dataloader
      （Full / Endpoint-only / No / Mismatched Context 四条件）

## 快速使用

```bash
cd ~/ManiSkill
python3 -m icl_pickplace.generate --num-pairs 16 --num-workers 8 --out-dir demos/icl_data
python3 -m icl_pickplace.validate --dataset demos/icl_data
python3 -m icl_pickplace.inspect_pair --dataset demos/icl_data --pair-id 0
```

环境依赖与排障：lerobot==0.4.4、numpy==1.26.4、torch cu128、系统 ffmpeg
（详见数据集内 `meta/icl/SCHEMA.md` 与 git 历史 README）。
