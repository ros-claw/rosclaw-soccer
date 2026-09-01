# S117：冲击恢复小脑、稳定性—可塑性闭环与通用 Growth 契约

## 结论先行

S117 没有把 S116 的一个成功反射包装成“端到端神经小脑已经学会”。本轮把真实扑球后的成功与失败轨迹编译成可复现课程，在四张 A6000 上依次验证动态增益肌肉记忆、失败优先 PPO、教师流形门、短时域反事实教师、后继状态目标、平滑低维动作计划、反馈学生蒸馏和失败前沿再训练。固定的获取集与保持集始终独立评分，任何候选只要没有新能力或破坏旧能力就被拒绝。

当前最重要的五项事实是：

1. 在早期同版本对照中，动态 `kp/kd` 记忆使零残差保持能力从 `90/128` 提升到 `107/128`，说明“肌肉记忆”必须包含控制器状态，不能只有关节目标。v5 默认控制器基线为获取 `7/128`、保持 `92/128`，它只可与使用同一默认控制器考试契约的学生比较。
2. 32 个失败状态、每状态 2 个扰动的鲁棒后继教师把组成功从 `1/32` 提升到 `12/32`，教师最大连续稳定步数中位数从 `0.5` 提升到 `14.5`；但三个反馈蒸馏学生仍未通过闭环门，最好的当前帧 ridge 也只有获取 `4/128`、保持 `91/128`。
3. 深度复核发现，历史 PPO `179200` checkpoint 的 `27/128、94/128` 曾错误地与默认控制器的 `7/128、92/128` 比较；前者把 ready 线速度阈值从 `0.18` 放宽到 `0.35 m/s`，角速度阈值从 `0.45` 放宽到 `0.90 rad/s`，并改变了目标 slew。这个 GPU 预选是不可比考试造成的假阳性，现已撤销。
4. 在相同候选控制器契约下重跑零残差 GPU 基线，成绩恰好也是获取 `27/128`、保持 `94/128`；新版选择器因此返回 `NO_CANDIDATE_QUALIFIED`。独立 CPU MuJoCo 使用新的 128 + 128 成对初态再次得到获取 `31→31/128`、保持 `92→92/128`，神经残差净成功数提升为 0。
5. 新增逐状态教师组合后，有效状态从单源最多 20 个扩到 26 个，location-head warm-start 首次越过预设门；65,536 步在线 PPO 在同规则 GPU 考试中取得获取 `32/128`、保持 `94/128`，相对零残差为 `+5/0`。这是首次神经净增益，但仍低于预设的获取 `+8` 晋级门；再加 65,536 步 failure-frontier 续训仍为 `32/128、94/128`，因此两个候选都被拒绝，当前仍没有资格进入球队全链路或 SIM Champion。

这是一轮从“识别假成长”推进到“得到小幅真增长但守住晋级门”的阶段：数据驱动训练、教师组合与在线 PPO 已跑通，神经策略第一次在同规则获取集超过冻结肌肉记忆 5 个 episode，同时零保持遗忘；但预先定义的最低增益是 8，不能事后降门。工程突破是 Growth 选择门既不奖励阈值漂移，也不把小幅改善夸成 Champion；不能据此宣称传球—射门—扑救系统已经变强。

## 问题定义

S116 的规则反射能修复一个密封邻点，但更难邻点仍失败。S117 不再继续增加角度补丁，而是训练一个只依赖可部署本体感的恢复策略：

- 输入：重力方向、根部线/角速度、朝向、关节位置/速度、上一目标、教师跟踪误差、足部状态和动态 `kp/kd`；
- 输出：29 关节 PD 目标周围的有界残差；
- 权限：只在教师流形之外获得连续可塑性，且最多作用有限恢复窗口；
- 目标：在历史失败获取集上产生连续 ready 状态，同时保持历史成功集；
- 边界：仅 `SIM_ONLY`，无硬件权限、无策略自晋升权限。

本阶段所称“成功”不是视频看起来站住，而是 pelvis、upright、线速度、角速度、双足支撑和连续稳定步数共同通过环境定义。

## 数据闭环

### 1. 因果失败快照

课程直接读取 S113—S116 的物理轨迹和评分结果。成功 episode 只提供教师记忆；失败 episode 只提供获取重置，禁止把失败动作偷偷当教师。快照从接触后的真实 `qpos/qvel` 建立，控制目标从下一控制帧开始，避免把已经消费过的命令重复一次。

课程逐步扩充为：

- v3：下一帧因果 PD 目标；
- v4：初始及全过程动态 `kp/kd`；
- v5：成功教师的因果 `qpos/qvel` 流形；
- v6：可选相位检索实验。

正式 v5 课程：

- [课程清单](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-curriculum-v5/impact-recovery-curriculum.json)
- manifest hash：`sha256:4353e643b1579d7059ca4ecdebc68204881309ea8bd4086f9cc6ac5da57a5a47`
- archive hash：`sha256:e192766cec86b68d8415898e1e6f8440900f7eba168eebc39213af912aea4061`

### 2. 获取与保持分离

每一个候选都必须参加相同种子、相同样本数的两场考试：

- `ACQUISITION`：只从历史失败分布重置，测是否学会新恢复；
- `RETENTION`：只从历史成功分布重置，测是否遗忘旧能力。

预选门要求获取至少比零残差基线多成功 `8/128`，保持最多下降 `4/128`。复核后新增一条更基础的要求：基线与候选的 episode、噪声、目标 slew、记忆模式以及全部 ready/success 阈值必须完全相同。v5 默认控制器基线 `7/128、92/128` 仍可供默认控制器学生使用；PPO `179200` 的匹配基线则为 `27/128、94/128`，两组不可交叉选择。

- [v5 获取基线](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-dynamic-gain-acquisition-baseline-v5-v1/diagnostic.json)，report hash：`sha256:982e738ceb4eef5ebe3517189ea6d544f97465dfb88b41cfaebd9d28cde595f2`
- [v5 保持基线](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-dynamic-gain-retention-baseline-v5-v1/diagnostic.json)，report hash：`sha256:c553496e72a9287948bbf02fb17c6164ae51866fa5117b72eacdf75ae2af6a90`

PPO 匹配规则复测：

- [匹配获取基线](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-matched-mid179200-acquisition-baseline-v1/diagnostic.json)，`27/128`，report hash：`sha256:9dce9f9bc925da37545c4a46ce375034ce7f71620156fa3a74b68f5a7e1ba344`
- [匹配保持基线](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-matched-mid179200-retention-baseline-v1/diagnostic.json)，`94/128`，report hash：`sha256:e4887a0fd60691cda4a6ebab2751178b75166961ffefd823124193c7bf5b9f7d`
- [新版匹配选择](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-matched-selection-v2/selection-report.json)，决策 `NO_CANDIDATE_QUALIFIED`，report hash：`sha256:225457b8b57135f4e663447b251545965c604edd738ef8ba2b2cbd3301fdf135`

## 实验复盘

### 实验 A：动态增益肌肉记忆

固定增益的成功记忆无法复现原 controller 的刚度/阻尼时序。把每帧 `kp/kd` 写入轨迹与课程后，零残差保持成绩：

| 记忆形式 | 保持成功 | 变化 |
|---|---:|---:|
| 关节目标 + 固定增益 | `90/128` | 基线 |
| 关节目标 + 动态 `kp/kd` | `107/128` | `+17`，即 `+13.3 pp` |

- [动态增益诊断](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-dynamic-gain-memory-diagnostic-v1/diagnostic.json)
- report hash：`sha256:ebf91ac128a0033f933b252330a10e46810bfabcd867ec44f066015801761057`

这项改善来自数据语义修复，不是手工给某个球路加动作。它说明小脑记忆必须保存“姿态 + 速度 + 控制器状态”的联合轨迹。表中 `90/107` 是早期同代码版本的配对消融，不能与后续环境契约下的候选交叉选择；正式选择使用上面的 v5 `7/92` 基线。

### 实验 B：状态最近邻相位检索失败

曾尝试让失败状态在成功轨迹附近搜索更相似相位：

| 相位半径 / 偏移惩罚 | 固定获取成功 |
|---|---:|
| 关闭相位搜索 | `26/128` |
| `1.0 s / 0.05` | `7/128` |
| `0.4 s / 0.20` | `7/128` |

结果表明局部姿态相似不等于控制因果相同。相位跳转破坏触球后命令时序，因此 v6 保留为可审计失败实验，默认继续关闭。

### 实验 C：平均稳定性 CEM 教师是假突破

四卡 CEM 从真实失败状态搜索短时关节残差：

- 40 个状态全部满足局部成本改善；
- 中位成本改善 `43.8%`；
- 但把计划放回完整 250 步固定获取考试，只成功 `12/128`。

也就是说，降低平均后退、侧移和角速度并没有直接学到“连续 25 步 ready”。这一轮证明平均稳定成本与真实终点资格不对齐。

### 实验 D：后继状态 + 平滑低维教师取得局部突破

教师目标随后显式加入：

- ready 帧占比；
- 最大连续稳定步数；
- 是否曾达到 25 步连续成功；
- 动作幅值与 slew；
- 身高、直立和有限状态安全约束。

同时把 80 步逐段独立动作压缩成 4 个时间结点，再线性插值到 16 个控制块，减少高频抽搐和单状态过拟合。

| 教师 | 教师状态成功 | 中位最大稳定步数 | 最大稳定步数 |
|---|---:|---:|---:|
| 零残差基线 | `3/40` | `0` | — |
| 平滑后继状态 CEM | `17/40` | `19` | `25` |

- [40 状态教师报告](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-successor-smooth4-40state-v1/teacher-report.json)
- report hash：`sha256:f3bc846b3feda6cbb6697d21040b4fd42be41320a17e946519e6f2f4453b194f`

这是局部教师真正改善成功定义的证据，但还不是可部署反馈策略。

### 实验 E：独立噪声暴露泛化断层与特权上限

把上述特权计划按课程行放回固定独立重置噪声：

- 获取成功 `17/128`；
- 比同源 v5 动态记忆零残差 `7/128` 高 `10`，但明显低于精确优化状态的 `17/40` 成功密度；
- 它使用课程行与控制相位的特权查表，没有未知状态反馈和保持集考试，因此只是教师上限诊断，不是候选策略。

- [计划库固定考试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-successor-smooth4-plan-bank-eval-v1/plan-bank-exam.json)
- report hash：`sha256:e133ca5c278cea43fd7a00746c3f3bd72a8e8301c3344eb8072ee9f505f86bcb`

根因很明确：每个计划只针对一个随机扰动优化，它记住了该次 reset，而不是该失败状态附近的恢复规律。

### 实验 F：反馈蒸馏仍未跨过闭环门

为避免开放环计划只适合一个 reset，本轮把本体感历史映射到 29 关节残差。大 MLP 在留出状态上比零动作更差；低容量当前帧 ridge 学生则把留出动作损失从 `0.10214` 降到 `0.06782`，相对改善 `33.6%`，因此获准参加固定考试。

- [蒸馏报告](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-successor-distilled-ridge-v1/distillation-report.json)
- report hash：`sha256:207be7021f8ec3cbc3c29130e191fdd74c3c9e8dc6d8e6a3632d7171487dcb18`

固定闭环成绩：

| 策略 | 获取 | 保持 | 相对零残差结论 |
|---|---:|---:|---|
| v5 动态记忆零残差参考 | `7/128` | `92/128` | 同课程、同代码、同种子基线 |
| 后继教师 ridge 学生 | `3/128` | `92/128` | 获取 `-4`，保持不变，拒绝 |

- [固定考试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-successor-distilled-ridge-eval-v1/evaluation-report.json)
- report hash：`sha256:0d2fa298df12e1ba0aa2df8543bb98894b71c80d2c14535792059f3380648525`

这个结果揭示了监督损失的局限：动作更像单状态教师，并不保证长时动力学更好。选择器按闭环成绩拒绝，而不是按 loss 晋升。

- [机器选择报告](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-distilled-selection-v1/selection-report.json)
- 决策：`NO_CANDIDATE_QUALIFIED`；report hash：`sha256:8390f321eac149e30c65b7047149e26334c04c6258ab368ec1e157c3ea9011a6`

## 针对根因的新改造：多扰动鲁棒教师

S117 后半段新增 `robust_variants_per_state` 和 `robust_worst_case_weight`：

1. 先选择内容绑定的不同失败课程行；
2. 对每一行生成多组独立关节位置、根部速度和关节速度扰动；
3. 同一套低维平滑计划在所有扰动上同时回放；
4. CEM 用“均值 + 最坏情形”联合成本排序；
5. 身高与直立约束取各扰动最坏值，成功要求组内全部扰动成功；
6. 蒸馏时重建相同扰动组，但每个变体独立决定是否产生训练标签；
7. 留出切分仍按课程行，禁止同一状态的不同噪声泄漏到训练和验证两侧。

这使教师搜索从“记住一次 reset”转为“在一个状态邻域里找共同恢复动作”。当前它仍是训练基础设施，不自动获得执行或晋升权限。

第一轮 8 状态 × 4 扰动试验中，零残差没有任何一组能让 4 个扰动全部成功；鲁棒共享计划有 `2/8` 组全部达到成功，其中最优组达到完整 25 步稳定，中位组仍为 0。该结果比单扰动“某一次成功”更严格，但样本很小且尚未参加独立固定考试，只能作为继续扩展的信号。

- [多扰动教师试验](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-successor-robust4-pilot-v1/teacher-report.json)
- report hash：`sha256:bef1a7e9c4c3180be25ef3fdc0125170abc02e09948cdce6065c83bdfd38f73c`

加入逐扰动身高/直立回归约束后，同 seed 严格复跑仍为教师 `2/8`、基线 `0/8`，中位成本改善 `39.05%`，说明该信号不是由不同扰动的平均值互相掩盖产生。

- [严格多扰动教师](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-successor-robust4-strict-pilot-v1/teacher-report.json)
- report hash：`sha256:547138ac632dd86bb3a156dc8a65e9cec237f91ab3a883789fd29cc52781de13`

但是只用 8 个课程行蒸馏反馈学生仍然失败：6 个训练行、2 个整行留出，ridge 的留出损失 `0.09234`，零动作损失仅 `0.03819`，相对改善为 `-141.8%`。该学生的 `student_exam_eligible=false`，没有参加固定闭环考试。

- [鲁棒教师小样本蒸馏](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-robust4-distilled-ridge-v1/distillation-report.json)
- report hash：`sha256:9e4c14a150e0773da86308acc355051be40d33a036822f8a29f51910c857e296`

这一反例把下一步从“调回归器”转为增加状态覆盖：32 个不同失败课程行、每行 2 个扰动，并要求至少 10% 的组完成后继成功后才允许蒸馏。

### 扩大到 32 × 2 的鲁棒教师

扩大后的教师不再只依靠 8 个状态判断方向。32 个不同失败课程行各生成 2 个独立扰动，同一计划必须同时面对该状态邻域。结果如下：

| 指标 | 零残差 | 鲁棒教师 |
|---|---:|---:|
| 组成功 | `1/32` | `12/32` |
| 最大连续稳定步数中位数 | `0.5` | `14.5` |
| 最大连续稳定步数上限 | — | `25` |

32 组全部满足成本与逐扰动安全约束，中位成本改善 `57.69%`，因此 `supervised_warm_start_eligible=true`。这是教师覆盖与后继状态上的实质提升，但仍是带特权的离线教师，不是可部署策略。

- [32 × 2 鲁棒教师](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-successor-robust2-32state-v1/teacher-report.json)
- report hash：`sha256:9e29fa9368f4a1f0938249d6bcc4d68d6a4ec6a325f17700dc412dfc237689d9`
- corpus archive hash：`sha256:51963c13fcc2193c0c8346ad6f3ef616b3b7ed4b7db37b33fb6ce084587f5448`

### 防泄漏蒸馏仍没有学会闭环动力学

蒸馏标签改为 `SUCCESSOR_FRONTIER`：只有教师真正成功，或至少连续稳定 15 步的变体才允许成为正标签；短暂降低成本但依然快速失败的变体只保留为反例。超参数选择只看内部整状态 calibration split，最终留出状态只考试一次，不再用外部 holdout 挑 ridge 正则或 MLP checkpoint。

| 学生 | 留出 loss 相对零动作 | 固定获取 | 固定保持 | 结论 |
|---|---:|---:|---:|---|
| 当前帧 ridge | `+21.95%` | `4/128` | `91/128` | 闭环获取 `-3`，拒绝 |
| 四帧历史 ridge | `+18.11%` | 未考试 | 未考试 | 未达留出门 |
| `64×64` MLP | `-15.37%` | 未考试 | 未考试 | 未达留出门 |

当前帧 ridge 的正则 `1000` 来自 3 个内部 calibration 状态；4 个外部留出状态从未参与选择。它的监督指标虽然通过，却在严格闭环中失败，再次证明“拟合教师动作”不能替代物理考试。

- [当前帧 ridge 蒸馏](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-robust2-32state-frontier-ridge-v1/distillation-report.json)，report hash：`sha256:4a622e4d364ef39f8a646dccc8be3a0854972cfdbf69f0f88ad3ca86f4f0c6d6`
- [当前帧 ridge 固定考试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-robust2-32state-frontier-ridge-eval-v1/evaluation-report.json)，report hash：`sha256:3f94bcdd1bc3626d41c5fb8fdeacbd6cc5d4c00cf8712b71c359b39a183f15c9`
- [当前帧 ridge 选择](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-robust2-frontier-selection-v1/selection-report.json)，report hash：`sha256:d721fcd9475b60b8717d06171fd3896d2f15c5ab6146796090148a0409a651a3`
- [历史 ridge 蒸馏](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-robust2-32state-frontier-ridge-history-v1/distillation-report.json)，report hash：`sha256:67fc0943a6cd604908e08fb4afb73bb4bfb70c1944de5365799fdd4065be6d6f`
- [MLP 蒸馏](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-robust2-32state-frontier-mlp64-v1/distillation-report.json)，report hash：`sha256:ab2e7a19fca7c43b77236f6aadd33cc65422b1288999955e5f2bd09c925df033`

### 历史 PPO “突破”经独立复核后撤销

`179200` 步中点模型曾得到获取 `27/128`、保持 `94/128`，旧选择报告把它与 v5 默认控制器的 `7/128、92/128` 比较并返回 `CANDIDATE_READY_FOR_CPU_FULL_CHAIN_EXAM`。进一步审计训练配置后发现，两边成功定义并不相同：

| 考试契约 | v5 默认基线 | PPO `179200` |
|---|---:|---:|
| ready 线速度上限 | `0.18 m/s` | `0.35 m/s` |
| ready 角速度上限 | `0.45 rad/s` | `0.90 rad/s` |
| 单周期目标 slew 上限 | `0.035 rad` | `0.010 rad` |

按 PPO 的同一控制器契约重跑零残差后，基线与候选完全相同：

| 同规则策略 | 获取 | 保持 | 神经残差净增 |
|---|---:|---:|---:|
| 零残差动态记忆 | `27/128` | `94/128` | — |
| novelty-gated PPO `179200` | `27/128` | `94/128` | `0 / 0` |

因此旧 selection hash `sha256:d07d54630a846228010cb8beeec34294583d0033ec9421314de43a9363bdb0d4` 只保留为历史反例，不能再作为资格依据。新版 v2 选择报告返回 `NO_CANDIDATE_QUALIFIED`。checkpoint 本身仍保持内容绑定，hash 为 `sha256:d5c19f2ad12b0aa355e2658ed984bd0dc58acbd316801ea5c389b88b631727fb`，但内容绑定不等于能力改善。

### 独立 CPU MuJoCo 成对复试

新增 `impact_recovery_cpu_exam.py`，不用 MJX rollout 标签，而是在普通 CPU MuJoCo 中重建同一套 29 关节 PD 控制器。每个初态的完整 `qpos/qvel`、课程行、seed 和 episode index 写入不可覆盖的 `scenario-suite.npz`；零残差 incumbent 与神经 challenger 必须逐样本使用完全相同的初态。

全量独立套件结果：

| population | CPU 零残差 | CPU PPO | 成对 rescue / regression |
|---|---:|---:|---:|
| acquisition | `31/128` | `31/128` | `0 / 0` |
| retention | `92/128` | `92/128` | `0 / 0` |

候选确实改变了动力学轨迹，但平均有效 residual RMS 只有 acquisition `0.00743`、retention `0.00325`；它没有让任何失败 episode 跨过连续 25 步 ready 门。CPU 决策为 `CANDIDATE_ARCHIVED_BY_CPU_MUJOCO`。

- [CPU 成对复试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-cpu-exam-128-v1/cpu-exam.json)
- report hash：`sha256:71aaae8ae91793fcdf538098a96de6db0be163101a7193544d0ff1f323ccdbc8`
- scenario suite hash：`sha256:d2b3e9ef2f96ecf5150c34da6af70750923d4e77ec372c0dcedf8c9be99a7b9d`

### 教师 warm-start 修复动作塌缩，但状态覆盖仍不足

CPU 复试显示 PPO 的有效 residual RMS 远小于鲁棒教师。新增 `impact_recovery_ppo_warm_start.py`，把 PPO actor 与鲁棒教师的时序反馈连接起来：

- critic、观测归一化器和分布 scale head 冻结；
- 19 个有合格后继标签的课程状态按整状态切分为 11 个 training、4 个 calibration、4 个 sealed exam；
- calibration 只用于选择 step，sealed exam 最后只计算一次；
- 输出仍只是在线 RL 初始化，不具有 physics exam 或晋升权限。

三种可塑性范围的结果：

| actor 可训练范围 | calibration loss 改善 | sealed exam 改善 | exam 动作 RMS | 决策 |
|---|---:|---:|---:|---|
| 全 actor | `5.05%` | `4.13%` | `0.1341` | 拒绝 |
| tail-128 + location | `7.30%` | `4.91%` | `0.1535` | 拒绝 |
| 仅 location head | `8.17%` | `6.76%` | `0.1150` | 拒绝 |

相比 parent exam 动作 RMS `0.00684`，warm-start 已解决“几乎不出力”的塌缩；但预先设定的 calibration 改善门是 `10%`，三者都没有通过。不能在看到结果后把阈值降到 8%。这把下一瓶颈定位为教师状态覆盖，而不是继续增加优化步数。

- [最佳 location-head warm-start](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-ppo-warmstart-location-v1/warm-start-report.json)
- report hash：`sha256:47ca6eaf4386d89a8ba03fde7cf0eb9a9a35474319f0c80be34a55b7ba253ce3`
- `warm_start_eligible=false`，未启动后续在线 PPO，也未参加 physics qualification。

### 教师组合打破状态覆盖瓶颈，并产生首个小幅物理净增益

固定预算扩展到 40 状态 × 2 扰动后，教师报告本身不能被简单称为更强：接受成本改善的状态为 `40/40`，但组成功率从旧教师的 `12/32`（`37.5%`）变为 `13/40`（`32.5%`），中位连续稳定步数从 `14.5` 下降到 `11.5`。它的价值是补充不同失败状态，而不是替代旧教师。

- [40×2 教师](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-successor-robust2-40state-v2/teacher-report.json)，report hash：`sha256:c0f25b970751406e72d21a1af92f6f8a833f9de8a3cf68509bf5d78c9b879624`
- 单独蒸馏只有 20 个有效状态，密封改善 `14.86% < 20%`，因此 `student_exam_eligible=false`；report hash：`sha256:7bdda018841738dce9d0d54e35cb3a92f5c8056ff0841b8da943d2bd8987799d`。

新增 `impact_recovery_teacher_portfolio.py`，不把两个教师的冲突动作直接混合。对每个课程状态按以下预注册顺序只选一个源：合格扰动数、成功扰动数、最小连续稳定步数、最小成本改善；完全同分时固定选择较低 source index。整状态选择完成后再切 training / calibration / sealed exam，重叠状态不会跨集合泄漏。

组合结果：

| 指标 | 旧单源 | 新单源 | 逐状态组合 |
|---|---:|---:|---:|
| 有效独立课程状态 | `19` | `20` | `26` |
| 训练行数（含 calibration 前） | — | — | `3280` |
| sealed loss 改善 | `21.95%` | `14.86%` | `28.65%` |
| 资格 | 通过单源蒸馏门 | 拒绝 | 通过 portfolio warm-start 门 |

- [教师组合报告](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-teacher-portfolio-v2/portfolio-report.json)
- report hash：`sha256:14075cb0bc3a4fc75b25e12680ee6fba753900945ebc4048d796f8e9d6c1390f`
- corpus hash：`sha256:2e09cb0027cfbf0980b2df01411aea54b9e16dc10b9d05b5fa68ec509c0e48b3`

同一 parent、同一 location head、同一 1000 步和学习率下，组合语料把 warm-start calibration 改善从 `8.17%` 提升到 `13.06%`，sealed exam 保持正改善 `2.81%`，动作 RMS 从 parent 的 `0.01291` 提升到 `0.17240`。这是第一次满足预设 `10%` warm-start 门。

第一次用 Soccer 环境 Brax 0.14.2 保存的 checkpoint 不能被 OpenTrack Brax 0.12.3 加载，错误为归一化状态多出 `std_eps`。实现因此取消 `load_config` 等版本私有 API，直接读取公开 `ppo_network_config.json` 重建网络，并在报告中绑定 Brax/JAX/JAXLIB/Orbax/Flax 序列化版本。用 PPO 原生环境重跑得到同一数值结论：

- [可续训 warm-start](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-ppo-warmstart-portfolio-location-opentrack-v4/warm-start-report.json)
- report hash：`sha256:2844b7483cd563be353bf78b57deaaf287f20037f017f5faca7bbde53ea973fe`
- warm checkpoint hash：`sha256:dbd39df044454047c7a6ba1fadc313fe18bba903c9c1d5a25010eae3f145ceae`

随后进行 65,536 步低学习率在线 PPO，并在与零残差完全相同的 128 + 128 固定种子考试上评估：

| 同规则候选 | 获取 | 保持 | 获取净增 / 保持下降 | 决策 |
|---|---:|---:|---:|---|
| 零残差动态记忆 | `27/128` | `94/128` | — | incumbent |
| portfolio warm + PPO v2 | `32/128` | `94/128` | `+5 / 0` | 拒绝（要求 `+8`） |
| 再加 65,536 步 failure-frontier PPO v3 | `32/128` | `94/128` | `+5 / 0` | 拒绝，无新增收益 |

- [v2 在线训练](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-portfolio-warm-ppo-v2/training-report.json)，report hash：`sha256:4d57568ff1fa00a70c2a5abfa9b2e61364cc8dc71e1d04024e7894c6e7c5b450`
- [v2 固定考试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-portfolio-warm-ppo-eval-128-v1/evaluation-report.json)，report hash：`sha256:dc0f71cb44011f1f22eecf65b85c5fefef907a5f93830a7070eecf551e22fc13`
- [v3 failure frontier](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-portfolio-frontier-v1/impact-recovery-frontier.json)，manifest hash：`sha256:0d81cdf29ba3b3f4b9113972008aa4348e68ea8b9b2de4c6fb979fdcc189979f`
- [v3 固定考试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-portfolio-frontier-ppo-eval-128-v2/evaluation-report.json)，report hash：`sha256:38518ffdc9ef90d9d68903b751fa1c2ef1d4b4b48d6fc63208245034847bd908`
- [双候选选择报告](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-portfolio-selection-v2/selection-report.json)，决策 `NO_CANDIDATE_QUALIFIED`，report hash：`sha256:a7b5ab67c6cd84c605ee0f7223c8a557c73385e975ec5c609a0d611594e39e9f`

这个结果是当前稳定性—可塑性闭环的准确边界：组合教师解决了动作塌缩和部分状态覆盖，在线 PPO 第一次提供 `+5` 个真实 acquisition 成功且零 retention 遗忘；但失败前沿再训练没有继续提高，说明剩余瓶颈不是“步数太少”，而是策略容量、反事实奖励与状态覆盖的一致性。门槛保持 `+8`，因此不启动 CPU 冠军考试、不接入球队全链路。

### 失败前沿续训仍没有产生神经净增益

系统曾依据旧预选择，从该模型 128 个获取 episode 中提取 43 个失败行，形成内容绑定的 capability frontier；其中触球后 `0—1 s`、`3—4 s` 和 `5 s+` 三个区间成功数都是 0，优先权由难度、近期失败和历史锚点共同决定。frontier 数据本身仍有效，但其“已通过预选”的上游资格已被本次复核撤销。

- [失败前沿](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-novelty-gated-mid179200-frontier-v1/impact-recovery-frontier.json)
- manifest hash：`sha256:c2193c17995e7e3159091f804a6751ab491d33816d3d125f287566a5e613f904`

续训降低学习率并混合失败前沿与保持回放。内部 16 环境评估的获取成功率从 `25%` 先跌到 `6.25%`，在 `115200` 步回升到 `37.5%`；但固定 128 + 128 考试中，最佳 checkpoint 仍只有获取 `27/128`、保持 `94/128`，与 incumbent 完全相同。

- [失败前沿续训报告](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-novelty-gated-frontier-consolidation-v1/training-report.json)，report hash：`sha256:228bd66e23dfd1cf7bfc36c0a6de42d6e0b16fa8fda91d1444f27828fe902c2a`
- [续训固定考试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-novelty-gated-frontier-consolidation-mid115200-eval-v1/evaluation-report.json)，report hash：`sha256:35065e2db5ae3113558ffb948ee843094dfaa41e4d6ef0abccf2a4ee12a2717b`
- [Champion 挑战](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-frontier-consolidation-champion-challenge-v1/champion-challenge.json)，决策：`CHALLENGER_ARCHIVED`；report hash：`sha256:bfe1390b3cf7ee38458a88baee5c4b4198ba5d9008a0e7cb0704cb2a582de63c`

这一步很关键：局部训练曲线变好不能直接覆盖 Champion；相同成功数也不能说明神经策略提供了净贡献。只有相同课程、相同种子、相同 episode、相同控制器考试契约下的获取目标实质改善，并同时满足保持 guardrail，challenger 才能前进。

## ROSClaw Core 通用化

足球代码只负责生成领域观测、MuJoCo 环境和实验报告。下列能力被抽象到 `rosclaw.continual`，不含 G1、球门、射门角度或足球字段：

- `FailureCurriculum`：失败优先课程及内容闭包；
- `IndividualScope`：策略、身体和个体记忆隔离；
- `LearnerBackend`：训练后端能力声明；
- `PlasticityLease`：有限时、有限范围的学习权限；
- `TeacherPrior` / `TeacherManifoldGateContract`：冻结教师和连续新颖度门；
- `SuccessorStateContract`：以可验证后继状态定义成长；
- `ResidualAdaptation`：残差策略的权限和回退边界；
- `SafetyProfile`：仿真、shadow 与真实边界声明；
- `ReproducibilityClosure`：数据、身体、配置、代码和模型哈希闭包；
- `ChampionRegistry`：候选与冠军分离，禁止训练结果自晋升；
- `PairedDominanceEvidence`：在内容绑定的同一考试套件上表达“目标必须改善、护栏最多有限退化”，tie 不能冒充成长。

这些契约只授权训练和证据记录，不提供电机接口，不允许 Agent 自批 REAL。

## 工程加固

本轮同时补齐：

- 动态 `kp/kd` 进入共享世界轨迹，课程和模型必须内容绑定；
- 非有限状态 fail closed；
- 教师、课程、模型、考试和选择报告逐层哈希绑定；
- 蒸馏考试独立验证每个 seed 的成功数、总数和成功率，不能手改汇总；
- MJX checkpoint 考试也重算逐 seed 汇总与 checkpoint 文件树哈希，Champion challenge 进一步要求该文件树真实存在于对应训练报告；
- baseline diagnostic v2 内容绑定完整 evaluation config 与 controller config；选择器 v2 提取 episode、噪声、目标 slew、记忆模式和 ready/success 阈值形成统一 exam contract，不同阈值直接 fail closed；
- 独立 CPU MuJoCo 复试保存完整初始 `qpos/qvel` 套件，重算逐 episode 成败、rescue/regression、力矩饱和、漂移和连续稳定步数，GPU 标签不能覆盖 CPU 物理结果；
- 未通过留出 loss 的学生不能参加固定考试；
- 后继状态教师除成本改善外还要达到显式组成功率，才能作为蒸馏热启动；
- successor 标签拒绝模仿短时降成本但快速失败的教师变体；
- ridge 与 MLP 只用内部整状态切分挑超参数，外部留出集只做最终考试；
- 多教师语料按完整课程状态只选一个鲁棒胜者，源报告、源 corpus、胜者表、组合 corpus 和 student model 逐层内容绑定；
- warm-start 不再依赖 Brax `load_config/_get_ppo_network` 私有 API，公开网络 JSON 与 Brax/JAX/JAXLIB/Orbax/Flax 序列化版本进入报告合同；
- warm-start 的最低 calibration/exam 改善不允许负值，退化模型不能靠配置进入在线 RL；
- 选择器同时识别 PPO checkpoint 和蒸馏学生，但二者使用同一获取/保持门；
- Champion challenge 重算内容绑定的 paired-dominance 证据，tie 与遗忘候选均 fail closed；
- 残差权限限制在教师恢复时域，超时自动归零；
- 所有新路径硬顶 `SIM_ONLY`、`promotion_authority=NONE`。

## 软件验证

- 本轮教师组合、warm-start、蒸馏、选择与 CPU 考试定向回归：`28 passed`；
- Soccer 全量功能回归（显式排除旧外部证据节点）：`762 passed, 15 skipped, 11 deselected`；
- 不排除时同样有 `762` 个功能测试通过，11 个历史外部证据节点因实现哈希变化按设计 fail closed；没有重写旧 JSON 冒充当前证据；
- 本轮 5 个变更 Python 文件通过 Ruff/format；教师组合与 warm-start 的 4 个相关源码/测试文件通过 mypy，compileall 通过；
- ROSClaw Core 的 continual + Practice + live ROS2 组合回归：`268 passed, 9 skipped`；Python 3.11 下 live ROS2 生命周期测试连续 10 次通过；
- 新增 Core/Soccer 文件的 Ruff、format、mypy 和 compileall 均通过；Core 分支基线的全 `continual` mypy 仍有 5 个既有错误，本次新增契约文件定向 mypy 为 0 错误；
- 通用 Core 改造已提交到 [rosclaw PR #485](https://github.com/ros-claw/rosclaw/pull/485)。

Soccer 全仓 format check 仍会报告 main 上 68 个既有文件与当前 Ruff formatter 版本不一致；Core 全仓也有既有 examples/benchmark lint 和 formatter 版本差异。本轮没有借机批量重排无关文件。

## 为什么本阶段没有新宣传视频

当前实验是隔离的触球后恢复子系统。新 PPO 在同规则 GPU 考试中取得 `+5` 获取成功且零保持遗忘，但仍低于预先设定的 `+8` 门，也尚未获得独立 CPU MuJoCo 冠军考试资格。现在挑一个好看的 seed 做宣传片，会把“小幅、未晋级的子技能增长”伪装成“小脑能力突破”。因此本阶段没有制作新宣传视频；只有候选先通过同规则成对门，再进入传球—射门—真实碰撞—扑救—摔倒/恢复整链路并通过后，才渲染阶段视频，像素始终不参与评分。

## 当前边界与下一步

当前已有数据驱动课程、四卡反事实教师、逐状态教师组合、反馈学生、可移植 warm-start、在线 PPO、稳定性—可塑性门、同规则双集考试和独立 CPU 成对复试；当前最好研究候选为获取 `+5`、保持 `0` 遗忘，但仍没有合格的神经小脑候选。下一步顺序是：

1. 保留当前 `+5/0` v2 checkpoint 作为研究回退点，不再从无增益 v3 继续堆步数；
2. 在线 RL 奖励改为同初态零残差反事实优势，并在 actor 更新中加入对 v2/retention 的显式行为锚定；failure frontier 仍须混合均匀与 retention replay，不能只放大近期失败；
3. 扩充教师组合时优先补当前 26 状态之外且在固定考试反复失败的状态，同时测量教师间动作冲突，不能只增加原始状态数；
4. 每个 checkpoint 先参加同规则 GPU 成对门；只有获取至少 `+8` 且保持下降不超过 4，才参加普通 CPU MuJoCo 成对门；任一层 tie 或遗忘都归档；
5. 只有隔离恢复通过最低净增益门，才接入传球—射门—碰撞—扑救—摔倒/恢复整链路；失败轨迹继续进入 failure frontier；
6. 整链路通过后再生成阶段视频并登记 SIM Champion；
7. 真实机器人仍需独立 body snapshot、permit、verified executor 和人工监督，本阶段不涉及。

S117 的准确结论是：**动态控制器记忆、鲁棒后继教师与逐状态教师组合取得了可复现提升；组合 warm-start 加在线 PPO 首次在同规则物理考试中得到获取 `+5`、保持零遗忘，但没有跨过 `+8` 闭环门，failure-frontier 续训也没有继续增长。跨后端复试修复了阈值漂移与 checkpoint 版本隐患；当前候选只保留为研究回退点，不能进入球队全链路。**
