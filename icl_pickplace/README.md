# ICL Pick-and-Place Paired Data Generator

基于 ManiSkill 3 生成 visual/action in-context learning 用的 **paired data**
（Context–Query 成对 demonstration），输出标准 **LeRobot v3** 数据集。

## 1. 采集了哪些数据

| 字段 | 内容 |
|---|---|
| `observation.images.agentview` | 固定外视角 RGB（h264 video 流） |
| `action` | 每步真实下发的关节位置指令 + 夹爪开合指令 |
| `observation.state` | 关节 qpos/qvel + TCP 位姿 |
| `timestamp` / `frame_index` / `episode_index` | 官方 v3 schema，20Hz 严格等间隔（已验证无漂移） |
| `meta/icl/`（辅助） | 物体 GT 位姿、抓取顺序、交互事件、成功标签、场景配置 |

规格：

- 480×360 @ crf18 默认（`--width/--height/--crf` 可调至 640×480 / crf15）
- fps = 20，磁盘约 25–40 MB / pair
- 视频是全数据集所有 episodes 连续编码的**单个 mp4**，
  由官方 `from_timestamp / to_timestamp` 按集索引切分（等效每集独立视频）

## 2. 数据结构与专家轨迹

目录结构就是标准 LeRobot v3（`data/`、`videos/`、`meta/info.json` 等），
官方 reader 直接可读，不再展开。在此之上约定了两个辅助文件（见 `meta/icl/SCHEMA.md`）：

- `pairs.jsonl` — pair → context/query 两个 episode 的显式映射
  （含 intended 顺序、两个 seed、验证状态）
- `episodes/episode_XXXXXX.json` — 每集的事件日志与 GT
  （抓取/放置事件、真实顺序、物体与容器规格）

专家轨迹来源（非遥操作）：

- 脚本策略调用 mplib 做 motion planning：
  RRT 做长距离避障路径，"screw" 螺旋运动做短距离精细接近
  （screw = 沿螺旋轴平移+旋转的直线式插值，用于垂直下探抓取这类精细动作）
- 所有 action 在物理仿真中逐步执行，抓取/放置均有物理验证
  （物体真实离桌并跟随 TCP；松爪后静止在容器内）
- **失败处理**：失败 episode 会如实标 `success=false` 并记录失败原因，
  所在 pair 默认**不写入**数据集（`--keep-failed` 可保留），
  失败统计汇总进 `report.json`——即"失败就不算数，但也不会被标成成功"

## 3. 任务设计

当前是最简 **object relocation（pick-and-place）**：

- 桌上 3–5 个物体 + 一个容器，Context demo 展示抓取顺序
- Query 换独立随机布局，按同一顺序执行
- 模型只需从 Context 推断顺序——无语言指令、无干扰物体（有几个抓几个）

Diversity 维度（每 pair 随机采样，Context/Query 共享物体集合）：

| 维度 | 取值 |
|---|---|
| 物体 | 数量 3–5；cube / box / cylinder；尺寸、HSV 颜色随机（同场景颜色可区分） |
| 容器 | `box` 开口盒、`round` 圆桶、`tray` 浅托盘、`tapered` 锥形桶（上宽下窄的程序化容器）、`ycb_bowl` 真实 YCB 碗 mesh |
| 桌面 | 拼色 / 木板条 / 圆桌 / 深色亮面 / 白色哑光 |
| 背景 | 纯色板 / 家庭 / 工厂 / 仓库 |
| 机器人 | panda（全容器）；xarm6_robotiq（实测 box、round，其余容器仍在调） |

## 4. 速度与成功率

（32 核 / RTX 4090，CPU 仿真 + 离屏渲染）

- **8 workers 即饱和**，16 workers 无增益（两档吞吐同为 ~0.17 pair/s）
- 两个数字的含义：
  - **系统吞吐**：8 workers 合计约每 6 秒产出一个 pair ≈ **600 pair（1200 episodes）/小时**
  - **单 pair 生命周期**：一个 worker 串行完成 context+query（含重试）约 35s；
    8 个 worker 并行错峰产出，整体仍等效 6s/pair
- 单次 episode 成功率约 75–95%（随容器类型 / 物体数波动）；
  带重试后 pair 有效率通常 ≥95%，好种子批次 16/16
- 参考批次：16 pair / 16 workers / 96.6s / 619MB

## 5. TODO

任务升级：

- [ ] Pose transformation（rotate / reorient / flip；
      变化初始姿态、目标角度、旋转轴、物体对称性）
- [ ] Constraint-aware manipulation（insertion、窄空间放置、绕障移动；
      例：方向键插销入槽、bagel 放进 toaster、开下层柜门）
- [ ] 长尾任务（推 toaster 控制杆、刮水器清洁台面）

难度与多样性：

- [ ] 语义级物体（需要类别识别，不只是颜色/形状）
- [ ] 冗余干扰物体（只抓一部分，考验 Context 指代能力）
- [ ] 相机视角 / 光照随机化；wrist camera；depth & segmentation
- [ ] paired 跨本体（同一顺序 demo 由不同机器人执行）

规模化与工程：

- [ ] **Scale up 大批量采集**：当前是 demo 级验证（几十 pair），
      需要千对级生产——分批生成、追加合并、磁盘与数据管理方案
- [ ] GPU sim 后端（ManiSkill GPU parallel）验证采集吞吐上限
- [ ] 拟人轨迹：当前 mplib 时间参数化偏机械，改 Hermite/quintic 插值
- [ ] 机器人池扩展（widowx、so100）；xarm6 支持剩余容器
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
（详见 `meta/icl/SCHEMA.md` 与 git 历史 README）。
