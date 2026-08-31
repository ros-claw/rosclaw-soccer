# S117：冲击恢复小脑、稳定性—可塑性闭环与通用 Growth 契约

## 结论先行

S117 没有把 S116 的一个成功反射包装成“端到端神经小脑已经学会”。本轮把真实扑球后的成功与失败轨迹编译成可复现课程，在四张 A6000 上依次验证动态增益肌肉记忆、失败优先 PPO、教师流形门、短时域反事实教师、后继状态目标、平滑低维动作计划和反馈学生蒸馏。固定的获取集与保持集始终独立评分，任何候选只要不会新能力或破坏旧能力就被拒绝。

当前最重要的两项事实是：

1. 在早期同版本对照中，动态 `kp/kd` 记忆使零残差保持能力从 `90/128` 提升到 `107/128`，说明“肌肉记忆”必须包含控制器状态，不能只有关节目标。随着环境契约继续加固，v5 当前严格基线重新测得获取 `7/128`、保持 `92/128`，后续选择只使用这组同哈希基线。
2. 后继状态 + 四结点平滑 CEM 教师在自己优化的 40 个失败重置上把成功从 `3/40` 提升到 `17/40`，特权计划库在固定独立噪声下为 `17/128`，比当前 v5 零残差获取高 10，但它按课程行查表且没有保持考试，不能成为策略；反馈学生固定考试只有获取 `3/128`、保持 `92/128`，没有学到新能力。

这不是一次“成绩很好”的阶段，而是一次把假进步挡在晋升门外、并定位到单状态优化泛化断层的阶段。针对该根因，代码已增加同一失败状态的多扰动共享计划和均值—最坏情形联合优化，作为下一轮鲁棒教师基础。

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
- `ChampionRegistry`：候选与冠军分离，禁止训练结果自晋升。

这些契约只授权训练和证据记录，不提供电机接口，不允许 Agent 自批 REAL。

## 工程加固

本轮同时补齐：

- 动态 `kp/kd` 进入共享世界轨迹，课程和模型必须内容绑定；
- 非有限状态 fail closed；
- 教师、课程、模型、考试和选择报告逐层哈希绑定；
- 蒸馏考试独立验证每个 seed 的成功数、总数和成功率，不能手改汇总；
- 未通过留出 loss 的学生不能参加固定考试；
- 后继状态教师除成本改善外还要达到显式组成功率，才能作为蒸馏热启动；
- 选择器同时识别 PPO checkpoint 和蒸馏学生，但二者使用同一获取/保持门；
- 残差权限限制在教师恢复时域，超时自动归零；
- 所有新路径硬顶 `SIM_ONLY`、`promotion_authority=NONE`。

## 软件验证

- S117 课程、教师、蒸馏、考试与选择定向测试 `27 passed`；
- Soccer 全量功能回归：`740 passed, 15 skipped, 11 deselected`；
- 原始全量同样有 `740` 个功能测试通过，另有 11 个历史外部证据节点因共享世界实现哈希变化按设计 fail closed；没有重写旧 JSON 冒充当前证据；
- 本轮变更文件 Ruff 通过；10 个全新源码/测试文件通过 format check；8 个相关源码文件通过 mypy；
- ROSClaw Core 的 continual + Practice：`276 passed, 4 skipped`，新增 22 个源码/测试文件 Ruff、format、mypy 通过；
- 通用 Core 改造已提交到 [rosclaw PR #485](https://github.com/ros-claw/rosclaw/pull/485)。

Soccer 全仓 format check 仍会报告 main 上 68 个既有文件与当前 Ruff formatter 版本不一致；Core 全仓也有既有 examples/benchmark lint 和 formatter 版本差异。本轮没有借机批量重排无关文件。

## 为什么本阶段没有新宣传视频

当前实验是隔离的触球后恢复子系统。最新学生在固定闭环门上明显失败，把它接回三台 G1 传球—射门—扑救链并挑一个好看的 seed，会把失败候选伪装成突破。因此本阶段没有制作新的宣传视频。只有候选先通过获取、保持和 CPU MuJoCo 整链路，视频才作为证据的下游可视化生成；像素不参与评分。

## 当前边界与下一步

当前已经有数据驱动课程、四卡反事实教师、反馈学生、稳定性—可塑性门和双集考试，但还没有通过晋升门的小脑模型。下一步顺序是：

1. 完成多扰动鲁棒教师的小规模试验，检查组内成功和最坏稳定步数；
2. 若有效，扩大课程行和扰动数，再蒸馏反馈学生；
3. 用完整固定获取/保持集筛选，拒绝监督 loss 好但闭环差的学生；
4. 通过后才进入 CPU MuJoCo 传球—射门—真实碰撞—扑救—摔倒/恢复整链路；
5. 整链路通过后再生成阶段视频；
6. 真实机器人仍需独立 body snapshot、permit、verified executor 和人工监督，本阶段不涉及。

S117 的准确结论是：**动态控制器记忆已经形成可复现提升，后继状态教师取得局部突破，但单扰动教师和反馈蒸馏没有跨过闭环门；新一轮已转向多扰动最坏情形鲁棒学习。**
