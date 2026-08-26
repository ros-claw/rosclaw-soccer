# S101：G1 侧向运动员专家——从“残差补丁”切换到可学习的加速—制动小脑

日期：2026-08-26
边界：`SIM_ONLY`；未连接 ROS/DDS/真机；视频像素不参与评分。

## 结论先行

S101 已完成一个独立、数据驱动、严格左右对称的 **Lateral Athlete Expert**。它不是再给旧门将加一个手臂或腰部标量补丁，而是把“目标距离、当前侧速、本体姿态、剩余时间、上一命令”映射为侧向加速/反向制动命令，再交给冻结的 29-DoF locomotion 小脑执行关节目标和力矩稳定。

最终 v5 候选完成 4×A6000、400 epoch 蒸馏，并在独立 CPU MuJoCo 的 16 条配对路线（左右各 0.5/1.0/1.5/2.0 m，两个扰动种子）中全部通过。最大端点误差由 S100 `sign-only full drive` 父代的 **5.461 cm** 降到 **2.635 cm**，候选平均误差由 **3.270 cm** 降到 **2.066 cm**；最低骨盆 **0.7535 m**、最低直立投影 **0.9942**、最大根角速度 **1.7748 rad/s**。进入 successor-state 窗口后，最差侧速只有 **0.001732 m/s**，最差根角速度只有 **0.004483 rad/s**。

这说明我们第一次把 S100 暴露的“门将根本到不了高远角”的问题，拆出了一个可训练、可哈希绑定、可独立考试的运动技能。但它还不是飞扑，也没有证明高远角扑救率已经提高；正确的下一步是把这个稳定侧移专家接到独立 Dive Expert，而不是重新把所有职责塞回一个 PPO actor。

## 为什么 S100 后不能继续调旧 PPO

S100 的 CPU/MJWarp 可达性审计已经把瓶颈定位清楚：

- 旧 residual actor 只能动侧向命令、腰和手臂，下肢足端策略仍由冻结 locomotion prior 控制；
- combat teacher 的下肢权重只有 25%，即使开满 drive/lunge，也无法在快速高远角球到达前同时完成足端换支撑、侧移和制动；
- 继续扩大 residual 会先放大角动量和跌倒率，而不是自然地产生运动员式并步/交叉步；
- 因此 S100 的正式结论是 `NEW_LATERAL_LOCOMOTION_DIVE_EXPERT_REQUIRED`。

S101 先解决其中可独立证伪的一半：**侧向运动员专家必须能在不同距离上加速、到点、刹住，并把一个低动量状态交给下一技能。**

## 数据、参考项目与采用边界

本轮复核了本地完整数据和参考模型：

| 来源 | 本地可用内容 | 本轮用途 | 为什么没有直接当物理真值 |
|---|---:|---|---|
| MotionDecode | Fencing Footwork 39 条 CSV；Two-Person Dodging 44 条 CSV | 动作类别与后续模仿先验候选 | CSV 是 G1 重定向运动学轨迹，但不自带本项目场景的接触、力矩、稳定 successor-state 证明；许可仅允许研究/原型并要求署名 |
| GR00T-WBC | G1 Walk ONNX；源码 Apache-2.0、权重 NVIDIA Open Model License | 第二低层小脑候选，完成许可和接口盘点 | 本轮没有把不同观察、关节和控制频率的模型未经物理资格考试就混进执行链 |
| RoboNaldo Deploy | 当前已在 ROSClaw CPU/MJWarp 门将链使用的 29-DoF locomotion TorchScript | S101 的冻结低层关节小脑 | 已有相同场景和关节映射；S101 检查点绑定其字节哈希，权重变化即 fail closed |

关键本地哈希：

- MotionDecode `README.md`：`a7c77f9c57cf9896f52a63d24b25148608695ca737dff2f0038e5b869e21ebe9`
- MotionDecode `LICENSE.md`：`ab46dc8e0994ea94fd6f66c0ec0cd33fa3d309702297cc5e220eda58bfc1d60e`
- GR00T-WBC Walk ONNX：`7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407`
- 本轮冻结 locomotion policy：`sha256:d1d91b0201beeb649a4624ba40052d10fe4aebe98bf6f4847decf75dd1fee2da`

所以 MotionDecode 确实“有帮助”，但正确用法是作为下一轮步态风格/足端相位模仿先验，再由 MuJoCo 重新学接触和稳定性；不能把人体或重定向 CSV 播一遍就宣称训练出了可用小脑。

## 新架构

### 1. 分层职责

`Lateral Athlete Expert` 是一个 11→96→96→96→1 的神经策略：

- 输入：侧向位置误差、侧速、误差/速度幅值、剩余时间、骨盆高度、直立投影、三轴根角速度、上一侧向命令；
- 输出：`[-1, 1]` 的局部侧向速度命令；
- 物理上限：乘以冻结 prior 已资格化的 `0.40 m/s`；
- 低层执行：冻结 29-DoF RNN locomotion policy 输出关节位置目标，2 ms MuJoCo PD/力矩上限执行；
- 权限：没有 ROS、DDS、硬件、晋升权限。

这比“端到端直接力矩 actor”保守，但它是当前证据允许的合理分层：先让独立技能真正学会加速和刹停，再扩大低层残差/力矩权限。否则高远角稀疏奖励会再次同时污染足端、手臂、飞扑和恢复。

### 2. 严格双侧对称

模型没有靠数据“希望”学出左右对称，而是对原始网络输出执行：

`command(x) = clip(0.5 * (f(x) - f(mirror(x))), -1, 1)`

因此镜像输入必然得到符号相反、幅值相同的命令。v5 全数据最大双侧对称误差为 **0**，CPU 配对路线的最大端点误差差距为 **0.8204 cm**。

### 3. 可学习的 accelerate–brake 教师

教师不是固定朝目标打满命令，而是：

- 位置误差负责加速；
- 当前侧速负责提前制动；
- 一个平滑 `tanh` 项补偿冻结 locomotion policy 的命令死区；
- 穿过零点的速度阻尼保持连续，避免在到点附近突然切换命令；
- 环境检测连续 0.30 s 进入 10 cm / 0.10 m/s 到点窗后，锁存释放命令，再检查后续 0.50 s 的 successor state。

输出使用物理边界一致的 `clip` 而不是 `tanh`。后者在满命令处只能无限逼近 ±1，v4 已实证产生系统性制动不足。

## 失败驱动的五轮闭环

| 候选 | 变化 | RMSE | 最大命令误差 | 结论 |
|---|---|---:|---:|---|
| v1 | 硬 deadband/最小命令切换 | 0.04336 | 0.32838 | 拒绝；教师不连续 |
| v2 | 平滑死区，但到点速度增益仍突然放大 | 0.04054 | 0.42936 | 拒绝；近零位置、高侧速样本断层更明显 |
| v3 | 去掉突变到点增益，连续速度阻尼 | 0.03170 | 0.19967 | 拒绝；满命令尾部仍欠拟合 |
| v4 | 输出从 `tanh` 改为物理一致投影，160 epoch | 0.01629 | 0.10952 | 拒绝；RMSE 过门，最大误差未过 0.08 |
| v5 | 同一架构继续 4-GPU 训练到 400 epoch | **0.003828** | **0.04502** | 拟合门通过，进入 CPU 物理考试 |

这里没有通过降低阈值“制造成功”。每个失败候选都保留在外部 evidence 目录，最终门仍是 RMSE ≤ 0.020、最大命令误差 ≤ 0.080、对称误差 ≤ 1e-6。

## CPU MuJoCo 物理结果

统一条件：原生 G1 stadium、10 s episode、0.02 s 控制、每控制步 10 个物理子步，即 **2 ms** 物理步；左右相同距离、相同种子扰动；球被停放，不参与技能评分。

| 指标 | S100 sign-only 父代 | S101 神经候选 | 结果 |
|---|---:|---:|---|
| 路线数 | 16 | 16 | 同分母 |
| 通过率 | 100% | 100% | 候选全部过独立门 |
| 安全率 | 100% | 100% | 无非有限、无关节越界 |
| 平均端点误差 | 3.2695 cm | **2.0658 cm** | 改善 36.82% |
| 最大端点误差 | 5.4610 cm | **2.6355 cm** | 改善 51.74% |
| 最低骨盆 | 0.7535 m | 0.7535 m | 无稳定性退化 |
| 最低直立投影 | 0.9942 | 0.9942 | 无稳定性退化 |
| 最大根角速度 | 1.6827 rad/s | 1.7748 rad/s | 仍低于 2.50 门；略高于父代，后续需继续压低 |
| successor 最大侧速 | 0.000213 m/s | **0.001732 m/s** | 候选更高但远低于 0.04 门 |
| successor 最大根角速度 | 0.003238 rad/s | **0.004483 rad/s** | 候选更高但远低于 0.20 门 |

候选到点更精确，但 2 m 右移的稳定窗最晚约 8.10 s，说明它目前是“稳定运动员基础”，还不是门将面对 0.3–0.5 s 高速射门时所需的飞扑。S102 必须增加短时 push-off/dive 专家，而不是把 S101 的长距离到点时间误报成高远角扑救突破。

## 代码与可复用模块

- `training/lateral_athlete_expert.py`
  - 内容绑定的 4-GPU DDP 蒸馏；
  - CPU/GPU 同构本体特征；
  - 连续 accelerate–brake 教师；
  - 严格左右等变神经解码；
  - 安全 `weights_only=True` 加载和 locomotion 哈希绑定。
- `training/lateral_athlete_cpu_exam.py`
  - 0.5–2.0 m 配对路线；
  - 2 ms CPU MuJoCo 真值；
  - 到点、端点、骨盆、直立、角动量、力矩、关节、successor-state 分门检查；
  - 评分报告与完整 qpos 轨迹分离。
- `media/lateral_athlete_video.py`
  - 只接受已通过 CPU 证据；
  - 视频、报告、轨迹三方哈希绑定；
  - `pixels_used_for_scoring=false`、`promotion_eligible=false`。

这些模块没有写死“足球得分”奖励，核心是通用的目标条件侧移技能、内容绑定和 successor-state 门；足球门将只是第一个实践场景。

## 证据与视频

证据根目录：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s101-lateral-athlete-expert-v5`

- 训练报告内部哈希：`sha256:e632cad85d1084f4f034a5d998c4faffe9046f18978b44e13fb8bece077a5d72`
- 候选 checkpoint：`sha256:d2e803d5bf7675aba5ba05b0f96e32564dbc4b9ce4563cbd3e6f32aa3bd26a63`
- CPU exam 内部哈希：`sha256:eb33baa0044387c1677604b07fb1ebf77dfb8f93fbad793685378eb9824d229a`
- 视频 manifest：`sha256:8fbf1c5e9ba5edadfcc780b41d1709715c1c111dfc4fb0a07619db8f1d67200d`
- 视频文件：`sha256:a0f5c341a282d906a36835dfa09e3e747574931e6fc6b864f7870f99b86d79f8`

阶段视频：[lateral-athlete-development-v2.mp4](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s101-lateral-athlete-expert-v5/lateral-athlete-development-v2.mp4)

视频为 1920×1080、30 fps、25.63 s，依次展示左右 2.0 m 和左右 1.5 m；标准足球只作为目标位置标记，存储的 G1 qpos 才是 CPU exam 的真实回放，视频没有重算或修改评分。

## 下一阶段：S102–S104

1. **S102 Dive Expert 数据面**：从 MotionDecode footwork/dodging、已有 targeted-dive、S101 push-off 前态中筛选短时侧蹬/腾空/落地窗口；逐条记录许可、G1 joint mapping、接触状态和来源哈希。
2. **S102 独立飞扑专家**：输入目标截点、到达时间、支撑脚和 S101 successor/pre-takeoff state；输出有界 29-DoF 轨迹或小残差力矩，训练时允许摔倒，但必须真实触球并进入可恢复终态。
3. **S103 技能路由器**：在 `Lateral Athlete / Dive / Absorb / Get-up` 间用环境拥有的状态机切换；网络不能自行绕过安全门。
4. **S104 无重置三 G1 闭环**：重接 S95 动态传球与 S96 上角射门，传球者、前锋、门将各自使用同一通用技能契约；成功/失败轨迹进入因果 replay，视频只从晋级后的 CPU 证据生成。

S101 的真正价值不是“门将横着走了 2 m”，而是把 ROSClaw Growth 从难以解释的一个大 actor，推进到 **可组合技能、独立训练、失败驱动、内容绑定、物理晋级、稳定交接** 的工程闭环。
