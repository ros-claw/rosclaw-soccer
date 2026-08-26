# S34：稳定运动底座的残差适应闭环实施报告

日期：2026-08-14
状态：`SIM_ONLY / REJECTED`
范围：ROSClaw 通用持续学习能力 + OpenTrack/G1 Athlete Foundation 实践
物理真值：CPU MuJoCo，50 Hz 控制，0.002 s 物理步长
训练资源：4 × NVIDIA RTX A6000（每卡 49,140 MiB）

## 1. 结论先行

S34 打通了一条真实、可重复、可否决的稳定性—可塑性闭环：

```text
冻结的稳定运动底座
        ↓
密封残差学习合同
        ↓
4 卡在线 PPO + 历史适应器 + 世界模型
        ↓
候选检查点 / ONNX
        ↓
冻结参数审计
        ↓
同动作、同身体、同物理父子考试
        ↓
保持集 + 新技能集 + 绝对物理门
        ↓
晋升 / 继续开发 / 拒绝
```

这不是“又写了一段 G1 脚本”。ROSClaw Core 新增的是与机器人、任务和训练框架无关的残差适应合同与参数隔离证据；OpenTrack/G1 只是第一个后端实践。

本轮获得了真实但有限的进步。最佳候选在四个新门将动作上的跌倒数从 `3/4` 降到 `2/4`，恢复数从 `1/4` 增到 `2/4`；新动作关节误差、关键点误差、jerk 和力矩饱和均改善，六个旧动作无跌倒且恢复率从 `5/6` 提升到 `6/6`。但是候选仍有两个新动作跌倒，绝对跟踪、脚滑、骨盆高度、力矩饱和、根角速度和转场门槛均未通过。

因此最终判定严格保持为：

> `REJECTED`

候选参数可以作为 S35 分析失败的研究材料，但该训练产物本身不再续训、不进入 champion、不允许硬件激活，也不能被表述成已经学会职业门将动作。除绝对物理不合格外，最终复查还发现实际训练步数超过了预先密封的合同上限；这一点在第 8 节详述。

## 2. 这一阶段解决了什么根本问题

### 2.1 从“足球补丁”转成“可成长的身体”

此前路线常把任务动作叠在站立/横移控制器上。手臂可以伸远，但腿、腰、支撑、腾挪、落地和恢复没有形成一个共同的运动底座，结果就是动作看起来僵硬，踢球或扑球以后还会摇晃。

S34 按《rosclaw_soccer讨论3》的方向，先暂停足球成功率优化，把目标收缩成一个更基础的问题：

> 在没有足球的情况下，稳定底座能否吸收左/右跳与左/右跨步，并且不遗忘已有跑跳能力？

这是 stability–plasticity dilemma 的最小真实实验：

- stability：已有跑跳动作不能因为学新门将动作而崩掉；
- plasticity：新动作不能始终只由旧模型做一个保守近似；
- safety：相对变好仍不等于绝对可用，必须保留物理资格门。

### 2.2 为什么不直接重训整个 29-DoF 网络

直接改整个网络最容易获得训练奖励，但也最容易把已经稳定的能力覆盖掉。本轮采用：

```text
applied_action = parent_action
               + scale × (candidate_action - parent_action)
```

父策略的继承隐藏层被冻结；候选只通过残差策略、历史编码器、世界模型和价值网络学习。残差幅度还要通过合同门控，不能因为训练 loss 下降就全量接管身体。

通俗地说，父策略像已经练成的基本功，adapter 像新学的专项肌肉记忆。专项记忆先以较小音量参与动作；只有同卷考试证明它既学到了新招、又没有毁掉基本功，才有资格逐步放大。

## 3. 数据与上游资产

### 3.1 固定的软件和父模型

- OpenTrack checkout：`cb9b751993a2483e5d1805a2565ddbfe950c04c9`
- 父模型：OpenTrack `specialist4` 官方 2B-step checkpoint
- 父 ONNX：`sha256:c7f42c251f9e416c697cf5b996f45c056e619dfec36a5a38bda43d98b936faba`
- 身体：Unitree G1 29-DoF
- 运行时：ONNX Runtime CPU + MuJoCo CPU

### 3.2 同卷考试动作

保持集共 6 段：

- `jumps1_subject1`
- `jumps1_subject5`
- `run1_subject2`
- `run1_subject5`
- `run2_subject1`
- `run2_subject4`

新技能集共 4 段：

- `leftjump`
- `rightjump`
- `leftstep`
- `rightstep`

每段轨迹都固定源哈希；父模型和候选模型使用相同 episode、起始帧、最大步数、身体和物理配置。

### 3.3 许可证复盘与修正

复查时发现早期 `opentrack-exam-plan-v2.json` 把四段重定向后的门将动作写成了 `rosclaw-research-generated`。这个标记不正确：预处理不会抹掉上游来源，它们继承 Humanoid-Goalkeeper 的 `CC-BY-NC-SA-4.0`。

本轮没有覆盖旧证据，而是：

1. 新建 `opentrack-exam-plan-v3.json`；
2. 把四段 acquisition motion 修正为 `CC-BY-NC-SA-4.0`；
3. 用 v3 重新运行父模型和候选模型的十段 CPU MuJoCo 考试；
4. 重新生成 promotion decision；
5. 在 evidence index 中把旧版本标记为 superseded。

六段 LAFAN 派生数据标记为 `CC-BY-NC-ND-4.0`。因此整套 S34 数据、模型与视频只能作为本地研究证据；不得进入商业 champion，也不应把派生数据提交进 MIT 代码仓库或对外再分发。

## 4. ROSClaw Core 的通用能力

新增 `src/rosclaw/continual/residual_adaptation.py`，包含两个核心对象。

### 4.1 `ResidualAdaptationContract`

合同在训练/考试前密封：

- 后端合同哈希；
- 父模型、身体、保持数据、新技能数据哈希；
- 冻结参数选择器和可训练参数选择器；
- 设备编号；
- 最大世界步数和学习率；
- rehearsal/acquisition 采样比例；
- 最大残差 RMS；
- 最大冻结参数漂移；
- `SIM_ONLY` 与禁止硬件执行边界。

合同执行严格字段校验、哈希校验、作用域不重叠校验、比例与数值有限性校验，并使用原子写入且拒绝覆盖。

### 4.2 `ParameterIsolationEvidence`

训练后证据检查：

- 是否仍引用密封父模型；
- 冻结参数前后哈希是否一致；
- 冻结参数最大漂移是否在合同内；
- 是否确实检查了冻结和可训练两个域；
- 残差输出是否在上限内；
- 实际训练世界步数是否在密封预算内；
- 保持集、新技能集和关键安全回归是否通过。

这个对象不认识 G1、足球或 OpenTrack，因此以后可用于机械臂抓取、移动机器人导航、无人机扰动适应等其他 ROSClaw Growth 后端。

## 5. Soccer/OpenTrack 后端实现

### 5.1 物理考试器

`opentrack_exam.py` 实现：

- 至少 8 个且保持/新技能均存在的密封考试；
- CPU MuJoCo 真正积分，不用视频像素或参考轨迹冒充物理结果；
- 父 ONNX 单输入与 adapter ONNX 的 `obs + 79-frame history` 双输入；
- 跌倒、恢复、关节 RMSE、root-relative MPJPE、脚滑、骨盆高度、力矩饱和、根角速度、jerk、转场误差；
- 父动作和候选动作逐步比较；
- raw residual 与实际施加 residual 分开记录；
- 非有限状态和证据覆盖均 fail closed；
- 输出目录必须在源码树外且拒绝覆盖。

### 5.2 稳定训练器

`opentrack_adapter_train.py` 为 OpenTrack AnyAdapter 增加了可复现的启动边界：

- 明确父 checkpoint；
- 明确保持/新技能动作及采样占比；
- 策略 LR、世界模型 LR、entropy、监督跟踪权重和最大步数均进入参数；
- 只允许 `SIM_ONLY`；
- 运行配置和证据路径不写入上游源码。

### 5.3 导出桥修复的两个真实上游问题

OpenTrack adapter checkpoint 的首次导出暴露了两个问题：

1. `config.json` 中序列化的 Python callback 字符串被重新灌入强类型 callback 字段；
2. trainer restore 后再读取环境 observation size，会触发 JAX `UnexpectedTracerError`。

`opentrack_adapter_export.py` 只移除 allow-list 中三个非网络 callback 字段，并记录输入/输出哈希；同时在 restore 前封存 I/O shape，再调用官方转换器。它没有更改任何网络数值字段。

候选 ONNX 的 JAX/ONNX 平均绝对误差为 `7.28e-7`，候选 ONNX 哈希为：

`sha256:046157e5817a9eff1d409e5bc090149ca17ee7ce806c0ef20df9b1a1656629f5`

### 5.4 参数隔离审计

`opentrack_adapter_audit.py` 直接加载父/子 Orbax 参数树，逐层比较继承隐藏层，并统计全部声明的可训练域。

| 域 | 参数数 |
|---|---:|
| 冻结父隐藏层 | 580,538 |
| residual policy adapter | 566,202 |
| value network | 642,305 |
| world model | 676,897 |
| history encoder | 111,168 |
| 可训练域合计 | 1,996,572 |

冻结参数最大漂移为 `0.0`，前后内容哈希一致。

### 5.5 晋升门

`opentrack_promotion.py` 要求：

- 父子 plan hash 完全一致；
- 候选 reference policy 必须是密封父模型；
- 冻结参数证明成立；
- 保持集无关键跌倒回归、总跌倒不增加、恢复不下降；
- 保持集关节误差限制在父模型的 120% 内，关键点、jerk、力矩饱和不得变差，最低骨盆高度最多下降 1 cm；
- 新技能集必须同时做到跌倒减少、恢复提高、关节/关键点/jerk/骨盆/力矩均改善；
- 实际训练步数不得超过合同的 `maximum_world_steps`；
- 仍必须通过独立的绝对物理门，才能 `PROMOTED`。

相对门回答“是否值得继续培养”；绝对门回答“是否已经够格”。两者不能混为一谈。

## 6. 两轮训练

### 6.1 第一轮：5.24M-step 纯 PPO adapter

第一轮完成 5,242,880 世界步并成功导出 ONNX，但全量残差使新技能集从父模型的 `3/4` 跌倒恶化为 `4/4`。0.25 倍残差虽出现改善信号，但训练本身缺少足够强的稳定性约束。

结论：拒绝全量候选，不晋升；它只用于指导第二轮超参数。

### 6.2 第二轮：20.3M-step stability-guided adapter

第二轮合同先于结果密封，主要配置为：

| 配置 | 数值 |
|---|---:|
| 世界步数 | 20,316,160（合同上限 20,000,000，**超出 316,160**） |
| 并行环境 | 4,096 |
| GPU | 4 × A6000 |
| rehearsal / acquisition | 60% / 40% |
| policy LR | 1e-5 |
| world model LR | 1e-4 |
| entropy cost | 0.001 |
| supervised tracking weight | 0.0002 |
| history | 79 × 93 |
| action | 29 |

运行耗时：JIT 390.86 s，训练 382.85 s。world-model loss 从 `112.7222` 降至 `65.1456`。训练 loss 只证明优化器在工作，不作为物理晋升依据。

OpenTrack 按并行 batch 对齐后多执行了 316,160 步。合同字段叫 `maximum_world_steps`，所以不能在训练完成后把这解释成可接受舍入。最终加固将 checkpoint 目录中的实际步数写入通用 `ParameterIsolationEvidence`，并在 `passes()` 中与合同上限比较；该项直接导致候选 fail closed。

## 7. 残差尺度扫描

在许可证修正前的同一数值动作集合上，候选依次以 `0.125 / 0.25 / 0.50 / 1.00` 倍残差运行 CPU MuJoCo。尺度扫描只用于选择后续同卷考试候选，不直接决定晋升。

| 残差尺度 | 保持跌倒 | 新技能跌倒 | 总恢复率 | 总关节 RMSE | 总 MPJPE | 总 jerk | 力矩饱和 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 父模型 | 0/6 | 3/4 | 60% | 0.12620 | 0.20971 m | 5,261.6 | 8.855% |
| 0.125 | 0/6 | 4/4 | 50% | 0.13475 | 0.20376 m | 5,925.2 | 10.220% |
| **0.25** | **0/6** | **2/4** | **80%** | 0.12982 | **0.19719 m** | **4,690.2** | **8.021%** |
| 0.50 | 0/6 | 3/4 | 70% | 0.15232 | 0.21219 m | 5,724.6 | 10.053% |
| 1.00 | 0/6 | 3/4 | 70% | 0.18674 | 0.19678 m | 4,779.5 | 8.673% |

0.25 是唯一同时减少新技能跌倒、提高总恢复并改善多项连续物理指标的尺度，因此进入许可证修正后的正式父子考试。

一个反直觉结果是 0.125 比 0.25 更差。原因是残差不是“越小越安全”：太小可能把候选的纠偏动作削弱到无法跨过动态支撑临界点。1.0 则过度改变父动作，明显提高关节误差。这说明 residual scale 本身也是需要考试选择的控制参数。

## 8. 正式同卷结果

以下来自修正许可证后的 canonical plan v3；数值与之前一致，但 plan/report/decision 哈希全部重签。

### 8.1 保持集

| 指标 | 父模型 | 0.25 残差候选 | 变化 |
|---|---:|---:|---:|
| 跌倒 | 0/6 | 0/6 | 无安全回归 |
| 恢复 | 5/6 | 6/6 | +1 段 |
| 离散成功 | 3/6 | 2/6 | -1 段 |
| 关节 RMSE | 0.08926 rad | 0.10296 rad | +15.3%，在 20% 门内 |
| MPJPE | 0.13934 m | 0.11921 m | -14.4% |
| jerk | 3,164.9 | 3,044.1 | -3.8% |
| 最低骨盆 | 0.55335 m | 0.54979 m | -3.6 mm |
| 力矩饱和 | 5.958% | 5.500% | 改善 |

这里必须特别说明：离散成功从 3/6 降到 2/6，不能隐藏。该 success 是多个绝对阈值的 conjunction，候选一段动作在边界附近跨线；相对保持门使用无跌倒、恢复和连续物理 guardrail 判断“没有灾难性遗忘”，但这仍是候选未达到绝对资格的重要原因之一。S35 应扩大保持集，并把 success 的置信区间/重复种子纳入门控。

### 8.2 新技能集

| 指标 | 父模型 | 0.25 残差候选 | 变化 |
|---|---:|---:|---:|
| 跌倒 | 3/4 | 2/4 | 减少 33.3% |
| 恢复 | 1/4 | 2/4 | 翻倍 |
| 离散成功 | 0/4 | 0/4 | 尚无合格动作 |
| 关节 RMSE | 0.19290 rad | 0.18325 rad | -5.0% |
| MPJPE | 0.33116 m | 0.32383 m | -2.2% |
| jerk | 8,656.3 | 7,486.1 | -13.5% |
| 最低骨盆 | 0.07798 m | 0.07869 m | 略改善，仍代表跌倒 |
| 力矩饱和 | 16.602% | 14.763% | -11.1% |

### 8.3 最终门控

- 相对保持门：通过；
- 相对新技能门：通过；
- 参数内容隔离：通过（580,538 个冻结参数漂移为 0）；
- 训练预算合同：失败（20,316,160 > 20,000,000）；
- 完整参数隔离证据门：失败；
- 绝对物理门：失败；
- 关键安全回归：0；
- 结论：`REJECTED`。

拒绝原因首先是 `training_steps_above_sealed_ceiling`，此外绝对门还失败于 tracking success、MPJPE、脚滑、骨盆高度、力矩饱和、根角速度和转场误差。换句话说，物理指标提供了有价值的学习信号，但该候选同时违反训练合同且未达到绝对资格，不能作为“可继续晋升的开发候选”。正确做法是从父模型重新发起一个预算能整除并行 batch 的新合同，而不是修改或宽恕既有合同。

## 9. 视频证据

正式视频：

`/code/rosclaw/rosclaw_football/evidence/s34-stable-foundation-adaptation-v1/g1-athlete-foundation-s34-rejected-v3.mp4`

视频属性：

- 42.9 秒；
- 1920 × 720；
- 50 fps；
- H.264；
- 24,609,099 bytes；
- SHA-256：`c6504dd4e1984bfbadedb3dd9b3ffda9d6bd1442b50362c6791a84485767cd96`。

开场页明确写出训练预算超限与最终 `REJECTED`。随后四段分别展示 left jump、left step、right jump、right step。每段：

- 左侧：0.25 倍 learned residual 在 CPU MuJoCo 中的实际运动；
- 右侧：同帧运动参考；
- 像素不参与 promotion；
- 标题明确写出 `3/4 → 2/4 falls`、`1/4 → 2/4 recovery` 与 `NOT YET CHAMPION`；开场最终审计结论优先于旧段落标题。

本视频没有足球是有意设计：S34 验证的是 Athlete Foundation，而不是用球门和特效掩盖底层身体问题。由于数据许可证含 NC/SA/ND 限制，该视频应视为本地研究展示，不应用于商业宣传。

## 10. 代码清单

ROSClaw Core：

- `src/rosclaw/continual/residual_adaptation.py`
- `src/rosclaw/continual/__init__.py`
- `tests/continual/test_residual_adaptation.py`

ROSClaw Soccer：

- `src/rosclaw_soccer/evidence/opentrack_exam.py`
- `src/rosclaw_soccer/evidence/opentrack_adapter_train.py`
- `src/rosclaw_soccer/evidence/opentrack_adapter_export.py`
- `src/rosclaw_soccer/evidence/opentrack_adapter_audit.py`
- `src/rosclaw_soccer/evidence/opentrack_promotion.py`
- `src/rosclaw_soccer/evidence/opentrack_video.py`
- `tests/test_s34_opentrack_exam.py`
- `tests/test_s34_opentrack_adapter_train.py`
- `tests/test_s34_opentrack_adapter_audit.py`
- `tests/test_s34_opentrack_promotion.py`
- `tests/test_s34_opentrack_video.py`

## 11. 验证结果

| 验证 | 结果 |
|---|---:|
| S34 Soccer 专项 pytest | 16 passed |
| Soccer 全套 pytest（复用 Core CUDA/Torch 环境） | 297 passed, 9 skipped |
| Core continual pytest | 65 passed |
| S34 Soccer ruff | passed |
| S34 Soccer mypy | passed |
| Core residual ruff | passed |
| Core residual mypy | passed |
| ROSClaw 精确入口 `python -m rosclaw.entrypoint --help` | passed |
| 4 × A6000 可见性 | passed |
| ONNX/JAX 数值一致性 | MAE 7.28e-7 |
| CPU MuJoCo 父子同卷复跑 | passed，候选被拒绝 |
| 硬件命令 | 0 |

首次在 Soccer 轻量 venv 运行得到 `275 passed, 30 skipped`，其中多数因该 venv 没有 torch。随后没有把它们留作“缺依赖跳过”，而是复用本机 Core 的 CUDA/Torch 环境重跑；加入预算超限回归测试后的最终结果为 `297 passed, 9 skipped`。剩余 9 项是旧 stacked-Core 合同、legacy parity 或明确要求额外合格资产的条件测试；本轮 S34 新测试没有跳过。

## 12. 证据索引

统一入口：

`/code/rosclaw/rosclaw_football/evidence/s34-stable-foundation-adaptation-v1/evidence-index-v1.json`

Canonical 关键哈希：

- plan hash：`sha256:17d5fde94ee70504ec2e6396abefdd727beacfb8dce337ea32bc1b9707418636`
- parent report：`sha256:70878ab162437a4ea58dc983137b8b9e7b048ab1e048e6ec51f07052a3ef3d3c`
- candidate report：`sha256:be02457dd72d98b86ea94baef750df0ee3951c93e3bc052b18a977f4a99bbcbb`
- isolation report：`sha256:4250f891c6c2bea941932c947d73c329bed8bebbc5f7f1a5b5ead40cadeb9f72`
- decision：`sha256:cf6d572a5a5c886c33b3a9e46b9d82d53693a8374f32eecb0a7f5b8c8c9f22c6`

旧版本没有删除，因为“证据不覆盖”本身是审计要求；evidence index 明确给出最低 canonical 版本。

## 13. 尚未解决的问题

1. 四个新动作仍有两个跌倒，0/4 达到绝对成功。
2. 最低骨盆约 0.079 m，说明失败 episode 仍彻底倒地，不是轻微晃动。
3. 脚滑约 1.32 m/s，离自然运动仍很远。
4. 离散保持成功少 1 段，虽然连续稳定性指标和安全指标通过相对门，也不能称为无代价学习。
5. 当前 acquisition 只有四段，动作覆盖和多样性不足。
6. 当前是 joint-position residual；还不是端到端安全力矩策略。
7. 尚未加入足球、来球感知、手套接触、扑救奖励或 second-save。
8. 数据许可只支持研究闭环，不满足商业模型或商业宣传要求。
9. 实际训练步数超过密封预算 316,160，暴露出 launcher 在提交训练前没有把并行 batch 对齐纳入预算检查。

## 14. S35 建议

### P0：先把 Athlete Foundation 做到绝对物理合格

- 引入 PHUMA/BONES-SEED 等更可靠且许可清晰的 G1 动作；
- 增加 landing、get-up、single-leg push-off、brake、cross-step 和 recovery；
- 对保持集与新技能集做多 seed、多起点和扰动物理考试；
- 将 success 置信区间加入 retention gate；
- 继续限制残差，先搜索 `0.20–0.35`，同时学习状态相关 residual scale，而不是固定放大。

### P1：门将 Motion Prior

- 将 left/right × step/jump/land/get-up 建成 position-conditioned skill latent；
- 使用 AMP/ASE 或等效 discriminator 约束动作分布，而不是只追关节 MSE；
- 从扑救中间姿态 reset，训练落地吸能和二次恢复；
- 只有物理合格 motion 才能进入专项先验库。

### P2：Whole-body Keeper RL

- Actor：proprio history + ball history + previous action；
- privileged critic：真实球速、落点、接触与完整物理状态；
- 奖励：true save、glove first-contact、coverage、recovery、second-save、motion quality；
- 惩罚：fall、landing impact、脚滑、根旋转、力矩与关节边界；
- 保持 joint-position residual，力矩直出只在有独立 safety critic、shadow rollout 和更强物理资格门后研究。

### P3：ROSClaw Growth / Dream

- Practice 失败自动进入 failure memory；
- DreamForge 围绕失败生成摩擦、时限、来球高度、落点、初始身体姿态等反事实场景；
- 训练 personal adapter，shared foundation 保持冻结；
- 用相同 retention/acquisition/absolute gate 考试；
- 个人能力跨角色稳定受益后，才进入 Social Consolidation 更新 shared foundation。

## 15. 对“成长”的通俗解释

这轮可以类比一个已经会基本跑跳的孩子学习门将侧扑。

- 旧身体完全冻结：他不会因为练侧扑就忘了怎么跑。
- 小 adapter 学专项：像增加一组新的肌肉记忆，而不是重写整个大脑。
- 保持集：考试他原来会不会的跑跳。
- 新技能集：考试左/右跨步和跳扑。
- 参数隔离：确认老师真的只改了允许改的作业本。
- 绝对物理门：即使比昨天好，只要仍会摔、脚滑或动作过猛，就不能发职业证书。

本轮从“4 个新动作倒 3 个”进步到“倒 2 个”，而旧动作 6 个都没有跌倒。这说明学习方向第一次出现了受约束的正向信号；但离“自然、灵活、真正扑球”还有明确距离。ROSClaw 的价值不是把这点进步包装成成功，而是把每次练习的父模型、数据、训练权限、失败、考试和晋级都变成可审计的成长链。
