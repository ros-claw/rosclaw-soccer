# S21–S24：G1 解剖学手扑、MotionDecode 蒸馏与 SONIC v1.1 闭环报告

日期：2026-08-14
范围：`SIM_ONLY`；4×NVIDIA RTX A6000；无真实机器人命令。

## 结论先行

本轮让门将的双臂、手腕和真实手套进入在线物理强化学习，并修复了一个会夸大成绩的评价漏洞：过去只要球撞到机器人任意部位就可能被视频称作扑救，现在训练奖励、GPU 留出集、CPU 考试和视频选段均区分“身体封堵”与“手/手套真实接触”。

S21 在 96 场 CPU MuJoCo 考试中相对冻结小脑把首扑率从 29.17% 提高到 33.33%，真实首手扑从 6.25% 提高到 9.38%，二扑从 5.21% 提高到 7.29%，零摔倒、零关节/力矩越界。它通过基线考试，但同种子仍未支配 S15 父代，因此只是安全的解剖学手扑分支，不替换 Champion。

随后真正把 MotionDecode 捕球/平衡/侧步/起身代理轨迹以 8% 低权重蒸馏进物理 actor。S23 在新的 96 场 CPU 考试中把真实首手扑从 5.21% 提高到 10.42%，二次手扑从 1.04% 提高到 3.13%，平均回报提高 3.66，根部角速度 P95 为 1.885 rad/s；但二扑总率由 9.38% 降到 7.29%，未达到预注册的“相对冻结小脑至少 +2 pp”要求，因此被拒绝，不能替代父代。S24 为补救二扑而增加选择权重，最终在新种子 GPU 留出集上回报下降且角速度尾部变差，也被拒绝。

因此，本轮得到的是**可信的局部突破**：数据先验确实让手参与更多，但尚未形成稳定、成熟的鱼跃门将。现役父代保持不变。

## 开发内容

### 1. 真正的双臂解剖学扑救语义

- G1 场景加入成人比例的 21×11×7 cm 椭球手套碰撞体；碰撞几何和可视几何一致。
- 接触分类按球所属 Body 与左右手/手套 geom 显式判断，球碰躯干、腿或髋不再算 `hand_save`。
- 多步 NumPy/Torch 奖励、MJWarp、CPU 考试和视频清单共同增加首/二次手扑率。
- 教师同时控制腰、双肩、肘和腕；修正经编译模型有限差分 Jacobian 验证过的肩/肘方向错误。
- 第二球前不再让双臂下垂归零，而是回到紧凑的门将准备姿态。
- 物理 PPO 按首手扑、二扑、二次手扑和回报综合选择与 rollout 精确绑定的 checkpoint。

这些能力虽然在足球门将案例中验证，机制本身是通用的：语义接触通道、Body 内容绑定、父代回放、稀有事件选择和 fail-closed 晋级门都可复用于抓取、接触操作或碰撞避障。

### 2. MotionDecode 不再只是“数据存在”

数据根目录现有 104,695 条 G1 CSV；完整足球子集 1,203 条：短传 302、长传 358、射门 152、控球 247、其他 144。数据为 G1 29-DoF、120 Hz。

本轮生成了内容寻址的 10 类门将代理库：ready、split-step、左右侧步、左右低扑代理、左右高伸手代理、中路封堵代理和起身恢复。每个条目绑定原 CSV 哈希、许可哈希、G1 Body 哈希、帧窗和质量指标，不把原始数据写入仓库。

限制必须保留：MotionDecode 没有门将鱼跃真标签，当前 catch/lateral/recovery 都是代理；许可仅允许学术研究、个人学习和非商业原型。因此它只能作为 `SIM_ONLY` 训练教师，不能支持“学会真人门将”的宣传结论，也不授权商业发布训练产物。

物理 PPO 中的用法是：保留因果任务空间教师和冻结的下肢 locomotion 小脑，只将代理轨迹的腰/双臂相对姿态以 8% 混合到预训练目标。选择一个完整人体瞬间，不逐关节拼极值；数据先验没有下肢控制权，也没有物理或硬件权限。

### 3. SONIC v1.1 更新与闭环 A/B

同步参考仓库至 `32c8260e54118b1f92b1fdeb9395d70d828e51a5`。新 v1.1 包含手腕姿态增强、按机器人航向归一化目标朝向、手腕朝向跟踪、足部加速度/能耗奖励与实时相机遥操作。ROSClaw Soccer 现在支持 `low_latency` 与 `sonic_v1_1` 两套不同输入/参考步长契约；v1.1 使用 5 帧参考间隔并做 heading normalization。

同一原生 MuJoCo 场景、100 控制帧的固定时域 A/B：

| 指标 | low_latency | SONIC v1.1 |
|---|---:|---:|
| 前进距离 | 2.050 m | 1.912 m |
| 最低骨盆 | 0.693 m | 0.703 m |
| 最大根部角速度 | 3.176 rad/s | 1.736 rad/s |
| 最大关节绝对角 | 1.566 rad | 1.511 rad |

v1.1 的固定时域稳定性更好，但前进距离较小；该 A/B 只证明“值得进入完整助跑—触球—恢复消融”，不授权自动替换当前跑动策略。

## 实验结果

### S21：标准手套、解剖学手扑基线

4 卡训练：2.56M physics world steps，256k policy samples。GPU 留出 256 场通过；CPU 96 场：

| 指标 | 冻结小脑 | S21 |
|---|---:|---:|
| 首扑 | 29.17% | 33.33% |
| 首手扑 | 6.25% | 9.38% |
| 恢复 | 29.17% | 33.33% |
| 二扑 | 5.21% | 7.29% |
| 二次手扑 | 1.04% | 1.04% |
| 平均回报 | 175.050 | 180.537 |
| P95 / 最大根部角速度 | 0.832 / 1.147 | 1.909 / 1.959 rad/s |

与同种子 S15 父代比较，S21 的二次手扑 +1.04 pp，P95/最大角速度分别降低 0.357/0.736 rad/s；但首扑、恢复、二扑和回报均小幅下降。因此父代门结论为 `RETAIN_PARENT_ARCHIVE_CANDIDATE`。

### S22–S24：MotionDecode 消融

| 分支 | 训练量 | GPU 留出 | CPU 96 场 | 结论 |
|---|---:|---|---|---|
| S22，8% 上肢先验 | 2.56M steps | 通过 | 首手扑 8.33→11.46%，二扑 5.21→6.25% | 二扑提升不足 2 pp，拒绝 |
| S23，二次手扑选择 | 3.84M steps | 新种子通过 | 首手扑 5.21→10.42%，二次手扑 1.04→3.13%，二扑 9.38→7.29% | 二扑回归，拒绝 |
| S24，二扑/手扑均衡选择 | 5.12M steps | 新种子拒绝 | 未进入 CPU | 回报 -0.67、峰值角速度 2.707，停止 |

S23 与相同 96 种子的 S15 父代比较：S15 首扑/恢复/二扑为 44.79/44.79/11.46%，S23 为 38.54/38.54/7.29%；S23 首手扑更高（10.42% 对 8.33%），但其余核心指标不足，父代保持。S21 在同一组种子上也没有支配 S15。

这组消融说明 MotionDecode 的代理捕球动作不是“没用”：它显著提高了真实手部接触；问题是它尚不能同时维持站位、身体封堵和第二球覆盖。这就是 Stability–Plasticity Dilemma 在本实验中的具体表现。

## 视频

### 安全解剖学分支（推荐看整体稳定性）

`/code/rosclaw/rosclaw_football/evidence/s21-regulation-glove-ppo-4gpu-dev-v1/g1-regulation-glove-save-safe-branch-s21-v1-1080p.mp4`

- 30 秒，1920×1080，30 FPS；
- 6 组 CPU 物理轨迹；
- 哈希 `sha256:598a9ad54f94652fe55093fda7fe9a5fc0fe0abff35954b7309139bd043038bd`；
- 视频注明父代 Champion 保留。

### MotionDecode 手扑开发分支（推荐看数据先验效果与不足）

`/code/rosclaw/rosclaw_football/evidence/s23-motiondecode-second-glove-ppo-4gpu-dev-v1/g1-motiondecode-glove-save-development-s23-v1-1080p.mp4`

- 30 秒，1920×1080，30 FPS，6 组左右/高低/连续扑救；
- 选自 CPU 考试轨迹，4/6 片段含二扑，所有片段至少含一次真实手扑；
- 哈希 `sha256:2b0cfcff1fdd04ca5f3517135ae715c033d680c060f00e46867676af42159e4d`；
- 永久标注 `DEVELOPMENT / CPU EXAM REJECTED / NOT PROMOTED`，不可误当 Champion 宣传片。

像素不参与评分；所有结论来自状态、接触、关节、力矩和父子配对数据。

## 证据索引

- S21：`/code/rosclaw/rosclaw_football/evidence/s21-regulation-glove-ppo-4gpu-dev-v1/`
- S22：`/code/rosclaw/rosclaw_football/evidence/s22-motiondecode-upper-body-ppo-4gpu-dev-v1/`
- S23：`/code/rosclaw/rosclaw_football/evidence/s23-motiondecode-second-glove-ppo-4gpu-dev-v1/`
- S24：`/code/rosclaw/rosclaw_football/evidence/s24-motiondecode-balanced-double-save-ppo-4gpu-dev-v1/`
- MotionDecode 代理库哈希：`sha256:7ece0599df369935fa11b13de868b5410b6bb22351a4e613cda334b2fcdf5dc9`
- SONIC A/B 报告哈希：`sha256:3e4651e082c4d2c76336a90a0abdcbfcfc11c626e74b6ad897525ebf8aef8ecb`
- S23 父代决定哈希：`sha256:7b028aa3fecd1c50c66a61b5f4394c6bfc1f2879c967724399ce43192f36e6c9`

## 下一阶段：怎样得到真正的鱼跃门将

1. 采集或生成有明确球轨迹、手接触和落地恢复标注的门将数据。普通 `Catching_Action` 只能提供上肢风格，不能教授起跳时机与落地卸力。
2. 在冻结小脑上增加受限的门将腿部 option：split-step、cross-step、push-off、侧落地和 get-up；高层 actor 只选 option，不直接夺取全部腿部力矩。
3. 用阶段化 actor-critic：第一扑、落地、回中、第二扑分别保留 replay 配额，并对二扑总数、手扑和角速度 CVaR 同时做 Pareto 选择。
4. 将手部目标升级为可微 task-space/Jacobian 残差，轨迹数据只提供姿态先验，球—手截点仍由因果观测决定。
5. SONIC v1.1 先做完整跑动—触球—恢复闭环 A/B；通过自然度、精度和稳定性三门后再替换球员 locomotion。

当前可以诚实地说：门将已经开始用手、能连续恢复，也有数据驱动证据；还不能说已经具备职业球员式飞身鱼跃。这个差距已被量化，下一轮不再靠扩大手套、挑镜头或降低门槛掩盖。
