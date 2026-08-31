# S117：冲击恢复小脑、稳定性—可塑性闭环与通用 Growth 契约

## 结论先行

S117 没有把 S116 的一个成功反射包装成“端到端神经小脑已经学会”。本轮把真实扑球后的成功与失败轨迹编译成可复现课程，在四张 A6000 上依次验证动态增益肌肉记忆、失败优先 PPO、教师流形门、短时域反事实教师、后继状态目标、平滑低维动作计划、反馈学生蒸馏和失败前沿再训练。固定的获取集与保持集始终独立评分，任何候选只要没有新能力或破坏旧能力就被拒绝。

当前最重要的四项事实是：

1. 在早期同版本对照中，动态 `kp/kd` 记忆使零残差保持能力从 `90/128` 提升到 `107/128`，说明“肌肉记忆”必须包含控制器状态，不能只有关节目标。随着环境契约继续加固，v5 当前严格基线重新测得获取 `7/128`、保持 `92/128`，后续选择只使用这组同哈希基线。
2. 32 个失败状态、每状态 2 个扰动的鲁棒后继教师把组成功从 `1/32` 提升到 `12/32`，教师最大连续稳定步数中位数从 `0.5` 提升到 `14.5`；但三个反馈蒸馏学生仍未通过闭环门，最好的当前帧 ridge 也只有获取 `4/128`、保持 `91/128`。
3. 对历史失败优先 PPO 的内容绑定中点 checkpoint 重新做当前严格考试后，获取达到 `27/128`、保持 `94/128`；相对当前 v5 基线是获取 `+20`、保持 `+2`，首次通过 GPU 预选择，获得进入 CPU MuJoCo 全链路考试的资格。它仍然不是正式 Champion，更不代表真实机器人授权。
4. 以这份候选为 incumbent 做失败前沿续训后，最佳 challenger 仍为 `27/128`、`94/128`。通用 paired-dominance 门要求获取至少再增加 1 且保持最多下降 1，因此 tie 被正确归档，旧候选保持不变。

这是一轮取得了严格预选突破、同时继续挡住假进步的阶段：现在已有一个数据驱动神经恢复策略可进入下一层考试，但还没有 CPU 整链路 Champion，也没有足够证据宣称完整传球—射门—扑救系统已经变强。

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

预选门要求获取至少比零残差基线多成功 `8/128`，保持最多下降 `4/128`。当前 v5 内容绑定基线为获取 `7/128`、保持 `92/128`。通过也只代表有资格进入 CPU MuJoCo 整链路，不代表晋升。

- [v5 获取基线](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-dynamic-gain-acquisition-baseline-v5-v1/diagnostic.json)，report hash：`sha256:982e738ceb4eef5ebe3517189ea6d544f97465dfb88b41cfaebd9d28cde595f2`
- [v5 保持基线](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-dynamic-gain-retention-baseline-v5-v1/diagnostic.json)，report hash：`sha256:c553496e72a9287948bbf02fb17c6164ae51866fa5117b72eacdf75ae2af6a90`

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

### 历史失败优先 PPO 首次通过严格预选择

监督学生失败后，本轮没有继续只调回归器，而是对先前 novelty-gated PPO 训练中的每个内容绑定 checkpoint 重新做当前严格考试。`179200` 步中点模型得到：

| 策略 | 获取 | 保持 | 相对 v5 基线 |
|---|---:|---:|---:|
| v5 零残差 | `7/128` | `92/128` | — |
| novelty-gated PPO `179200` | `27/128` | `94/128` | 获取 `+20`，保持 `+2` |

四个独立 seed 的获取成功分别为 `8/32、7/32、7/32、5/32`，保持分别为 `25/32、24/32、22/32、23/32`。这不是单 seed 偶然，也没有靠牺牲保持集换取获取。选择器因此返回 `CANDIDATE_READY_FOR_CPU_FULL_CHAIN_EXAM`；该词的准确含义只是“可参加下一层考试”，报告仍明确写入 `promotion_eligible=false`。

- [PPO 严格固定考试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-novelty-gated-scratch-mid179200-eval-v1/evaluation-report.json)
- evaluation hash：`sha256:533972df0d9f1fefc970e12345042cd3abef71ccf30c3c19d7ed56080b2ffe06`
- checkpoint hash：`sha256:d5c19f2ad12b0aa355e2658ed984bd0dc58acbd316801ea5c389b88b631727fb`
- [PPO 机器选择](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-novelty-gated-mid179200-selection-v1/selection-report.json)
- selection hash：`sha256:d07d54630a846228010cb8beeec34294583d0033ec9421314de43a9363bdb0d4`

### 失败前沿续训没有超过 incumbent

通过预选择后，系统从该模型 128 个获取 episode 中提取 43 个失败行，形成内容绑定的 capability frontier；其中触球后 `0—1 s`、`3—4 s` 和 `5 s+` 三个区间成功数都是 0，优先权由难度、近期失败和历史锚点共同决定。

- [失败前沿](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-novelty-gated-mid179200-frontier-v1/impact-recovery-frontier.json)
- manifest hash：`sha256:c2193c17995e7e3159091f804a6751ab491d33816d3d125f287566a5e613f904`

续训降低学习率并混合失败前沿与保持回放。内部 16 环境评估的获取成功率从 `25%` 先跌到 `6.25%`，在 `115200` 步回升到 `37.5%`；但固定 128 + 128 考试中，最佳 checkpoint 仍只有获取 `27/128`、保持 `94/128`，与 incumbent 完全相同。

- [失败前沿续训报告](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-novelty-gated-frontier-consolidation-v1/training-report.json)，report hash：`sha256:228bd66e23dfd1cf7bfc36c0a6de42d6e0b16fa8fda91d1444f27828fe902c2a`
- [续训固定考试](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-novelty-gated-frontier-consolidation-mid115200-eval-v1/evaluation-report.json)，report hash：`sha256:35065e2db5ae3113558ffb948ee843094dfaa41e4d6ef0abccf2a4ee12a2717b`
- [Champion 挑战](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s117-impact-recovery-frontier-consolidation-champion-challenge-v1/champion-challenge.json)，决策：`CHALLENGER_ARCHIVED`；report hash：`sha256:bfe1390b3cf7ee38458a88baee5c4b4198ba5d9008a0e7cb0704cb2a582de63c`

这一步很关键：局部训练曲线变好不能直接覆盖 Champion。只有相同课程、相同种子、相同 episode 数下的获取目标实质改善，并同时满足保持 guardrail，challenger 才能前进。

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
- 未通过留出 loss 的学生不能参加固定考试；
- 后继状态教师除成本改善外还要达到显式组成功率，才能作为蒸馏热启动；
- successor 标签拒绝模仿短时降成本但快速失败的教师变体；
- ridge 与 MLP 只用内部整状态切分挑超参数，外部留出集只做最终考试；
- 选择器同时识别 PPO checkpoint 和蒸馏学生，但二者使用同一获取/保持门；
- Champion challenge 重算内容绑定的 paired-dominance 证据，tie 与遗忘候选均 fail closed；
- 残差权限限制在教师恢复时域，超时自动归零；
- 所有新路径硬顶 `SIM_ONLY`、`promotion_authority=NONE`。

## 软件验证

- S117 课程、教师、蒸馏、考试、选择与 Champion challenge 定向测试 `34 passed`；
- Soccer 全量功能回归：`740 passed, 15 skipped, 11 deselected`；
- 原始全量同样有 `740` 个功能测试通过，另有 11 个历史外部证据节点因共享世界实现哈希变化按设计 fail closed；没有重写旧 JSON 冒充当前证据；
- 本轮变更文件 Ruff 通过；10 个全新源码/测试文件通过 format check；8 个相关源码文件通过 mypy；
- ROSClaw Core 的 continual + Practice + live ROS2 组合回归：`268 passed, 9 skipped`；Python 3.11 下 live ROS2 生命周期测试连续 10 次通过；
- 新增 Core/Soccer 文件的 Ruff、format、mypy 和 compileall 均通过；Core 分支基线的全 `continual` mypy 仍有 5 个既有错误，本次新增契约文件定向 mypy 为 0 错误；
- 通用 Core 改造已提交到 [rosclaw PR #485](https://github.com/ros-claw/rosclaw/pull/485)。

Soccer 全仓 format check 仍会报告 main 上 68 个既有文件与当前 Ruff formatter 版本不一致；Core 全仓也有既有 examples/benchmark lint 和 formatter 版本差异。本轮没有借机批量重排无关文件。

## 为什么本阶段没有新宣传视频

当前实验是隔离的触球后恢复子系统。虽然 PPO checkpoint 已通过 GPU 获取/保持预选择，但尚未通过 CPU MuJoCo 传球—射门—真实碰撞—扑救—摔倒/恢复整链路。现在挑一个好看的 seed 做宣传片，会把“子系统候选”伪装成“球队能力突破”。因此本阶段没有制作新宣传视频；整链路通过后再把严格轨迹渲染为阶段视频，像素始终不参与评分。

## 当前边界与下一步

当前已有数据驱动课程、四卡反事实教师、反馈学生、在线 PPO、稳定性—可塑性门、双集考试和一个通过 GPU 预选的小脑候选；仍没有通过 CPU 整链路与正式 Champion 门。下一步顺序是：

1. 把 PPO `179200` 候选装入 CPU MuJoCo 全链路，只读取其本体感并验证动作权限、动力学和确定性重放；
2. 在传球—射门—真实碰撞—扑救—摔倒/恢复链中同时检查获取与历史能力保持，失败轨迹继续进入 failure frontier；
3. 若 CPU 结果不一致，把 mismatch 保留为反例，不用 GPU 标签覆盖 CPU 物理真值；
4. 若整链路通过，再生成阶段视频并登记 SIM Champion；
5. 后续训练必须超过当前 `27/94` incumbent；tie、单 seed 变好和内部曲线变好都不能晋升；
6. 真实机器人仍需独立 body snapshot、permit、verified executor 和人工监督，本阶段不涉及。

S117 的准确结论是：**动态控制器记忆和鲁棒后继教师都取得了可复现提升，监督蒸馏仍未跨过闭环门；一个 novelty-gated PPO checkpoint 已首次通过严格 GPU 获取/保持预选择，但失败前沿续训只追平 incumbent，下一步必须由 CPU MuJoCo 全链路决定它能否成为 SIM Champion。**
