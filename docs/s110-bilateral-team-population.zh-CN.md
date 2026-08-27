# S110：双侧触球、扰动种群与可学习失败账本

日期：2026-08-27
边界：`SIM_ONLY`，CPU MuJoCo 物理裁判；没有 ROS/DDS/真实 G1 指令；视频像素不参与评分。

## 结论先行

S110 没有宣称“球队已经通过双侧连续扑救”。正式 population 的结论是：

- 4 个 case 全部完成两次严格复演，轨迹摘要逐 case 相同；
- 左、右解剖足都发生了真实 MuJoCo 足—球碰撞；
- 右脚控制、重球高摩擦、左脚前沿和镜像 lane 覆盖了 4 个质量、4 个草地摩擦值、2 个 lane；
- 0/4 case 通过完整连续链，因此 `promotion_status = REJECTED_BILATERAL_POPULATION`；
- 失败没有被丢弃，而是按“第一扑、rearm、第二前锋触球、第二扑、后继状态”拆成 Growth 相位账本；
- 当前数值栈下，右脚控制组也不能复现历史 S109 完整成功。纯净 `main@ad201b8` 得到同一失败轨迹，证明这不是 S110 双侧改造造成的回归，而是旧 evidence 未绑定外部 ROSClaw Growth Core 和数值运行时所暴露的可复现性缺口。

因此，本轮真正的突破不是做出一个更漂亮但无法复演的视频，而是把“历史成功退化、左脚接触不足、镜像场景碰撞”变成了可精确归因、可严格回放、不会误晋升的训练输入。

## 开发内容

### 1. 角色无关、左右脚可复用的触球栈

数据驱动触球 actor 过去默认使用右踝。现在运行时显式接收：

- 当前角色的 29 个 actuated DoF；
- 实际击球踝 body；
- 左右镜像符号；
- 足端击球点到球心的实测距离。

左脚在自己的实时雅可比上解码力矩；actor 仍在规范右脚任务空间工作，通过矢状面反射迁移到左脚。右脚路径不引入坐标变换，保留历史调用语义。

关节目标 residual 和直接 torque residual 也增加 `kick_foot`，左脚输出是右脚肌肉记忆的精确矢状面镜像。未知脚位、非法镜像符号、错误踝 body 都 fail closed。

### 2. 第二前锋的双侧物理契约

`G1PhysicalSecondStrikerConfig v2` 新增：

- `kick_foot`；
- 角色独立的 actor 接近半径；
- 左脚站位偏移镜像；
- 左右脚必须与最终观测到的解剖接触足一致。

共享世界结果升级到 v19，并记录第二前锋 actor 的足—球距离。默认 actor 接近半径为 `0.25 m`，与 S109 共享 actor 的有效包络保持一致；左脚前沿 case 在受约束范围内使用 `0.30 m`。

### 3. population 与失败归因

新增 `bilateral_continuous_team_population.py`。每个 case：

1. 从时间零构建四台 G1、两颗实体球；
2. 不使用 ball cannon、teleport、reset 或 qpos/qvel 热写；
3. 跑一次正式物理世界；
4. 用完全相同的输入再跑一次；
5. 同时比较 result、评价对象和 trajectory digest；
6. 保存内容哈希绑定的 NPZ 和 JSON；
7. 把失败门控映射到相位和下一学习角色。

种群配置强制至少四个 case，并且必须同时覆盖左右脚、至少两个 lane、至少两个球质量和两个草地摩擦值。任一 case 失败，整体不得晋升。

### 4. 修复“末端失败吞掉前段信用”

旧 `_continuous_metrics()` 要求第二前锋触球时间和门将二扑手套时间同时存在。门将如果没有碰到第二球，函数直接返回 invalid，于是已经发生的 actor 激活、目标 residual、torque memory、静止球和连续时钟也全部被误判失败。

现在按相位计算：

- 只要遥测结构有效，连续时钟就可独立评分；
- 只要第二前锋发生接触，就能计算触球前速度和三层接触栈活动；
- 门将手套时间缺失只使二扑相位失败，不擦除第二前锋已经获得的能力信用。

同一批 v2/v3 物理轨迹摘要完全一致，但 v3 能正确识别：

- `right-control`：rearm 与第二前锋触球相位通过；
- `right-heavy-grip`：第二前锋触球相位通过；
- `left-foot-frontier`：第二前锋触球相位通过；
- `right-mirrored-lane`：第二前锋触球相位仍失败。

这是 Growth 闭环所需的“阶段信用”，不是放宽整链成功标准。

### 5. 运行时依赖内容绑定

历史 S109 evidence 绑定了足球项目源码和训练资产，但没有绑定外部 ROSClaw Growth Core 文件及数值库版本。本轮 request 新增：

- `growth_core_contract_hash`；
- MuJoCo 版本；
- NumPy 版本；
- ONNX Runtime 版本。

正式 v3 运行时为：MuJoCo `3.11.0`、NumPy `2.3.5`、ONNX Runtime `1.28.0`。以后环境升级必须重新认证，不能把历史 evidence 自动视为当前可执行能力。

## 正式 population

| case | lane / 脚 | 球质量 / 草地摩擦 | 足—球接触 | 触后峰值速度 | actor 峰值 | 二扑 | 主要失败 |
|---|---|---:|---:|---:|---:|---:|---|
| `right-control` | left-inner / 右 | 0.41 kg / 0.10 | 752.756 N | 9.440 m/s | 63.367 Nm | 否 | 第一扑当前认证失败、二扑未接触、全局关节安全 |
| `right-heavy-grip` | left-inner / 右 | 0.46 kg / 0.16 | 685.309 N | 9.019 m/s | 59.389 Nm | 否 | 第一扑/rearm、二扑、最终 ready |
| `left-foot-frontier` | left-inner / 左 | 0.43 kg / 0.08 | 720.400 N | 7.391 m/s | 83.386 Nm | 否 | 第一扑/rearm、左脚关节边界、20 次机器人接触、二扑 |
| `right-mirrored-lane` | right-inner / 右 | 0.40 kg / 0.05 | 无 | 0 | 0 | 否 | 初始拓扑引发 781 次机器人接触，第二球触前速度异常 |

右脚控制 case 仍证明了若干保留能力：传球 `5.602 s`、第一前锋触球 `7.390 s`、第一次高位手套接触 `8.014 s / 1.418 m`、实测 rearm `16.780 s`、第二前锋右脚触球 `16.950 s`。但第一球最终仍越过门线，第二球也没有形成合格手套扑救，且世界级关节安全失败，所以不能沿用 S109 的成功声明。

左脚前沿 case 在 `16.900 s` 由左脚真实触球，触球前球速仅 `0.000518 m/s`，触后前向峰值 `7.081 m/s`。这证明双侧路由和肌肉记忆迁移已通，但左脚高球、踝安全和与其他智能体的空间解耦仍未学会。

## 实验中的一次主动否决

早期 population v1 把球—脚接触摩擦与球—草地摩擦错误绑定，导致控制组的草地摩擦从 `0.10` 变成 `0.05`。该结果只保留作开发诊断，不作为正式结论。v2 起将两套物理参数拆开：goal spec 保留球—脚材料，population 只扰动 `ball_ground_friction`；v3 在同一修正物理配置上加入相位信用。

## 证据与视频

正式 evidence：

- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s110-bilateral-continuous-team-population-v3/evidence.json`
- 同目录下四个 trajectory NPZ 和 `request.json`

正式开发诊断视频：

- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s110-bilateral-growth-diagnostic-v2/s110-bilateral-growth-diagnostic.mp4`
- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s110-bilateral-growth-diagnostic-v2/s110-bilateral-growth-diagnostic.json`

视频为 38.7 秒、1920×1080、30 fps、1161 帧。它依次展示当前运行时右脚控制和左脚前沿，永久标注 `REJECTED DEVELOPMENT · NOT PROMOTED`。视频文件、evidence、request、goal contract 和两条轨迹均有 SHA-256 绑定，像素不参与门控。

## 通俗解释

过去系统像只看整场比赛的比分：门将最后没扑到第二球，就说所有人都没学会，连第二前锋确实用左脚把球踢出去这件事也不承认。这样“失败”没有教学价值。

S110 把比赛拆成接力棒：第一脚射门、门将恢复、第二前锋触球、门将第二扑、全队收尾。整场仍然是失败，奖杯不给；但系统知道第二前锋已经拿稳了哪一棒、在哪一棒掉球、下次应该训练哪名球员。与此同时，左右脚、球重、草地摩擦和站位变化会主动制造困难，避免只背会一个固定镜头。

## S111 建议：先恢复当前运行时冠军，再扩大训练

下一阶段不应继续盲目扩大参数网格，建议按以下顺序：

1. 以当前绑定运行时重新认证第一射手—门将前缀，分别训练第一射门落点和第一次解围，恢复可复演的冻结控制组；
2. 用 v3 账本中的成功触球/失败飞行片段训练 target-conditioned contact actor v2，而不是继续手调 yaw/pitch/loft；
3. 左脚训练损失同时加入目标飞行速度、左踝软边界、机器人安全距离和后继站稳，避免只奖励球速；
4. 镜像 lane 先做初始拓扑可达性优化，再启动 RL，当前 781 次互撞不能作为有效足球训练；
5. 采用“一名角色可塑、其余角色冻结”的 alternating Growth，先修前缀，再修第二前锋，最后训练门将二扑；
6. 通过 current-runtime holdout 后，才恢复不带红色 rejected 标签的宣传视频。
