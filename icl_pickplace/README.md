# ICL Pick-and-Place Paired Data Generator

基于 ManiSkill 3 生成用于 visual/action in-context learning 的 **paired data**（Context–Query 成对 demonstration），输出标准 **LeRobot v3** 数据集。

## 1. 采集了哪些数据

| 字段 | 内容 |
|---|---|
| `observation.images.agentview` | 固定外视角 RGB（video 流，h264） |
| `action` | 每步真实下发的关节位置指令 + 夹爪开合指令 |
| `observation.state` | 关节 qpos/qvel + TCP 位姿 |
| `timestamp` / `frame_index` / `episode_index` | 官方 v3 schema，严格 20Hz（1/fps 等间隔，已验证无漂移） |
| `meta/icl/`（辅助） | 物体 GT 位姿、intended/actual 抓取顺序、交互事件、成功标签、场景配置 |

**规格**：480×360 @ crf18（默认，`--width/--height/--crf` 可调至 640×480/crf15）；fps=20；磁盘约 25–40 MB/pair。视频文件为全数据集 episodes 连续编码的单个 mp4，由官方 `from_timestamp/to_timestamp` 索引切分，不按 pair 分文件。

## 2. 数据结构与专家轨迹

目录即标准 LeRobot v3（`data/`、`videos/`、`meta/info.json`…），官方 reader 直接可读，不再展开。额外约定见 `meta/icl/SCHEMA.md`：`pairs.jsonl`（pair → 两个 episode 的显式映射 + 顺序 + seeds）、`episodes/episode_XXXXXX.json`（每集事件与 GT）。

**专家轨迹来源**：非遥操作。脚本策略用 mplib 做 motion planning（RRT 避障 + screw 精确到位），在物理仿真中逐 action 执行；抓取/放置均有物理验证（物体真实离桌且跟随 TCP / 松爪后静止于容器内），失败不静默标成功。

## 3. 任务设计

当前是**最简 object relocation（pick-and-place）**：桌上 3–5 个物体 + 一个容器，Context demo 展示抓取顺序，Query 换独立随机布局、按同顺序执行。模型只需从 Context 推断顺序——**无语言指令、无干扰物体（有几个抓几个）**。

**Diversity 维度**（每 pair 随机，Context/Query 共享物体集合）：

- 物体：数量 3–5、形状 cube/box/cylinder、尺寸与 HSV 颜色随机（同场景颜色可区分）
- 容器 5 类：box / round 桶 / tray 浅盘 / tapered 锥桶 / 真实 YCB bowl（mesh）
- 桌面 5 种：拼色 / 木板条 / 圆桌 / 深色亮面 / 白色哑光
- 背景 4 场景：纯色板 / 家庭 / 工厂 / 仓库
- 机器人：panda、xarm6_robotiq（后者暂限 box 容器）

## 4. 速度与成功率

实测（32 核 / RTX 4090，CPU 仿真 + 离屏渲染）：

- **8 workers 即饱和**（16 workers 无增益，0.166 ≈ 0.167 pair/s）
- 吞吐约 **600 pair（1200 episodes）/小时**，单 pair 平均 ~35s
- 单次 episode 成功率约 75–95%（随篮型/物体数波动）；带重试后 pair 有效率通常 ≥95%，好种子批次 16/16
- 例：16 pair / 16 workers 96.6s，619MB

## 5. TODO

任务升级：
- [ ] Pose transformation（rotate/reorient/flip；变化初始姿态、目标角度、旋转轴、对称性）
- [ ] Constraint-aware manipulation（insertion、窄空间放置、绕障；如方向键插销入槽、bagel 入烤炉、开柜门）
- [ ] 长尾任务（推烤面包机控制杆、刮水器清洁台面）

难度与多样性：
- [ ] 语义级物体（类别需识别，而非仅颜色/形状）
- [ ] 冗余干扰物体（抓一部分，考验 Context 指代）
- [ ] 相机视角 / 光照随机化；wrist camera；depth & segmentation
- [ ] paired 跨本体（同一顺序 demo 由不同机器人执行）

工程：
- [ ] 机器人池扩展（widowx、so100）；xarm6 支持全部容器
- [ ] **GPU sim 后端**（ManiSkill GPU parallel）验证采集吞吐上限
- [ ] **拟人轨迹**：当前 mplib 时间参数化偏机械，改 Hermite/quintic 插值
- [ ] 采集官方任务（PickCube/StackCube 等）作为对照数据
- [ ] train/val/test split 与 eval dataloader（Full / Endpoint-only / No / Mismatched Context 四条件）

## 快速使用

```bash
cd ~/ManiSkill
python3 -m icl_pickplace.generate --num-pairs 16 --num-workers 8 --out-dir demos/icl_data
python3 -m icl_pickplace.validate --dataset demos/icl_data
python3 -m icl_pickplace.inspect_pair --dataset demos/icl_data --pair-id 0
```

环境依赖与排障见 git 历史 README 或 `meta/icl/SCHEMA.md`（lerobot==0.4.4、numpy==1.26.4、torch cu128、系统 ffmpeg）。
