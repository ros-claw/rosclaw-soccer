# 条件化接球训练与新场景拒绝（2026-09-29）

判定：**未获得可宣传球队能力**。所有结果为 MuJoCo 共享 3v3 球场 SIM_ONLY。底层 RoboNaldo G1 locomotion 权重冻结；新动作只走已有 `TeamMotorTarget` 与有界 `NavigationDelta`，没有真实机器人操作。

## 从失败里学到了什么

1. 冻结脚端动作在六种邻近来球位置/速度上只有 `0/6` 安全触球，证明单点任务空间命中不泛化。见 `shared-world-foot-contact-and-generalization-2026-09-29.zh-CN.md`。
2. 第一轮八组有界导航参数搜索（消费过的四种来球）最稳妥方案 `front04` 为 `2/4` 安全脚触球、`1/4` 有效向前球速；`negative_lat08` 虽有两次较快回球，但另一组发生非足部碰撞，被整体门拒绝。证据 `/data/rosclaw_overflow/rsi-team-intercept-navigation-search-v15/report.json`，hash `sha256:7dcc8466f4f66ca416b45f3aabbb9b639c03472a5d011d9991abf9c379bc5041`。
3. 发现站位导航固定追踪左脚，而真实摆腿脚由步态决定。改为按实测双脚高度/横向误差选择当前可及脚后，四组候选中的最佳 `phase_front04` 在五种已消费来球里安全触球 `3/5`、有效回球 `2/5`；仍低于预设训练门。证据 `/data/rosclaw_overflow/rsi-team-phase-navigation-search-v16/report.json`，hash `sha256:fa09f4e72e03a06a2e841daa2d36287637a63ac218854928c79a68436d54d44b`。
4. 根据以上训练轨迹，冻结了一个只在入场第 30 帧读取当前足球 x 间隙的条件化导航：间隙 ≤1.4m 时启用横向修正，否则不启用。它在已消费五场达到 `4/5` 安全脚触球、`3/5` 有效回球，逐帧导航审计均通过，因而按预声明规则打开 fresh holdout。证据 `/data/rosclaw_overflow/rsi-team-context-phase-training-v17/report.json`，hash `sha256:a6654dfb8e182e53b3e3aaaf2ae92a6c0be9396bfae7c78968849c3ca5091804`。这只是训练域选择结果，不能当泛化结论。
5. 在事先冻结的八种全新来球上，与原版成对比较：Candidate 安全脚触球 `3/8`，Parent `0/8`；但 Candidate 有效向前回球 **`0/8`**，Parent `0/8`。预声明门要求 Candidate ≥`6/8` 安全脚触球、≥`4/8` 有效回球，并且各项严格优于 Parent；因此 **REJECTED**。所有 8 组无候选非足部碰撞，导航与足端动作审计通过。证据 `/data/rosclaw_overflow/rsi-team-context-phase-fresh-holdout-v18/report.json`，hash `sha256:b074859685b89dc7d1a4883c3f417c50d30136a58dfbc2909c78e7a02f88c332`。不得把训练域 4/5 当成有效晋升。

## 原因与下一关

原方案主要是改变触球前站位和脚的目标**位置**，没显式控制足端**速度、接触法线方向、击球时机**。新场景里偶尔碰到球，却通常是小幅拨动或方向不对。继续围绕 1.4m 门槛改补丁只会过拟合。下一阶段应收集跨场景的“球相对足位置/速度、足端速度、支撑腿状态、关节动作、球碰撞和球速变化”样本，训练输出受限足端目标速度/动作时序的策略，配备 Parent 对照、分布外拒绝、稳定性和非足部接触门。

为此已把**当前实测双脚线速度**加入只读 `TeamFootKinematics` 快照：来自 MuJoCo 当前雅可比乘 `qvel`，没有给电机任何可写模拟器对象或额外硬件权限。带合格 G1 asset 的 5 秒、6 机器人 no-op 对照证明打开该观测后所有世界物理轨迹逐数组精确相同；`tests/rsi/test_team_foot_kinematics.py` 共 4 项通过。这个是未来数据驱动击球小脑的输入桥，不是新策略已成功的证明。

下一轮实验的最低晋升顺序仍为：接球/回球训练域 → 预声明新场景 → 球队完整传接射扑 → 连续自主比赛 → 可宣传视频。当前停在第二步失败，真实机器人及团队能力均未晋升。
