# S57：候选漂移 DAgger、独立稳定门与失败关闭置信门

日期：2026-08-21

状态：`SIM_ONLY / 4 CANDIDATES REJECTED / NOT_DEPLOYABLE`

## 结论先行

S57 没有产生可保留或可宣传的 G1 候选，但打通了三项此前缺失的 ROSClaw 通用成长能力，并用四个完整物理反例确定了下一步瓶颈：

1. 候选不再只看父策略访问的正常状态；系统会让旧候选在正常路线中闭环运行，把它自己造成的漂移状态回灌为“零纠偏”反例；
2. 稳定性从加权总成本中拆成独立硬门，其他指标改善不能再掩盖身体更晃；
3. 神经学生增加“是否需要纠偏”置信门，并用潜变量训练分布包络对 OOD 状态强制零输出。

最有价值的两个反例是：

- DAgger-v1 把未见失败状态改善提高到 `1.8215%`，同时把 S56-v3 正常稳定赤字的 `+1.0278 pp` 恶化扭转为 `-0.0208 pp` 小幅改善，但正常总代价仍回退 `1.4809%`，方向门失败；
- confidence-gate-v1 让正常总代价改善 `1.5673%`、稳定赤字改善 `0.3542 pp`，但失败改善只有 `0.7796%`，未达到 `1%` 硬门，正常偏航也仍超容差。

这说明 S57 已把 Stability–Plasticity 的两端都分别做到，但尚未在同一候选上形成合格交点。所有候选都被正确标记为 `student_development_retained=false`；没有制作视频，也没有 CPU MuJoCo 晋升复考。

## 新增的通用 Growth 闭环

### 1. 正常路线 on-policy DAgger

S56 的正常零动作语料只来自父策略。一旦学生的小误差把状态推离父策略分布，网络不知道应当沉默。

S57 新增以下循环：

`父策略正常状态 50% + 最新候选访问状态 50% → 全部标零纠偏 → 源状态分层训练/留出 → 重新蒸馏 → 重新选择净有益通道 → 双物理考场`

每条路线保留 10 个父策略样本和 10 个最新候选样本。72 个训练源参与梯度和通道选择；24 个留出源只参加最终考核。被拒候选可以在 `SIM_ONLY` 中作为反例采集器，但其报告会明确记录源候选并未被保留，不能把“用于学习失败”误写成“能力已通过”。

### 2. 独立稳定性硬门

S56-v3 的正常总代价和三个方向门均通过，但稳定赤字从 `49.1667%` 增至 `50.1944%`。新规则对正常路线单独计算稳定容差：

`max(父稳定赤字 × 2%, 0.002)`

候选稳定赤字超过该上界即拒绝。失败状态考场则不允许稳定赤字有任何回退。新证据写入器进一步保证：即使调用方误把物理考场的总 `passed` 写成 `true`，只要独立稳定字段为 `false`，成长保留位仍强制为 `false`。

用新门回放 S56-v3 时，允许回退为 `0.9833 pp`，实际回退 `1.0278 pp`，因此会被明确拒绝。

### 3. 学习置信门与 OOD 失败关闭

置信门不是手写时间线。它使用纠偏 MLP 的 64 维本体潜变量，训练一个平衡逻辑分类器：

- 正类：教师纠偏失败轨迹；
- 负类：候选自己访问的正常/漂移轨迹；
- 输出：连续 `[0, 1]` 纠偏开度；
- OOD：潜变量超出训练集 99.9% 对角标准化距离包络时，开度直接置零。

在完全未见源的离线门检查中，失败状态平均开度为 `0.7317`，正常/漂移状态为 `0.2147`。但这些分类数字不是晋升依据；门必须真正进入关节增量闭环并经过物理考场。

## 四个候选的闭环结果

| 候选 | 活动通道 | 未见失败改善 | 失败稳定变化 | 正常总代价回退 | 正常稳定变化 | 主要拒绝原因 |
|---|---:|---:|---:|---:|---:|---|
| DAgger-v1 | 16 | `+1.8215%` | `-2.5000 pp` | `+1.4809%` | `-0.0208 pp` | 正常总代价与方向门失败 |
| DAgger-v2 | 14 | `+0.3433%` | `-0.6250 pp` | `-0.7676%` | `+0.5069 pp` | 可塑性不足、正常偏航门失败 |
| confidence-gate-v1 | 16 | `+0.7796%` | `-0.8333 pp` | `-1.5673%` | `-0.3542 pp` | 失败收益不足、正常偏航门失败 |
| gate-v2 / gain 0.22 | 14 | `+0.7055%` | `-1.0417 pp` | `+1.7158%` | `+2.0486 pp` | 正常成本、方向、稳定三门失败 |

负数正常回退表示候选优于父策略。稳定变化为候选赤字减父赤字，负数表示更稳定。

第四候选也给出重要反证：根据失败状态局部 Jacobian 裁掉偏航冲突通道 10、25，并提高剩余通道增益，在 600 步正常闭环中没有保持线性作用，反而重新放大横向和稳定漂移。因此后续不能继续用单状态 Jacobian 做长时通道扫参。

## 外部证据

### DAgger-v1

`/code/rosclaw/rosclaw_football/evidence/s57-corrective-student-dagger-v1/normal-dagger50-maxchannel16-gain016-seed5600/student-report.json`

- 报告哈希：`sha256:35fd84d64bb57881fcc268a1a033a69ec9731f2aaca9563fbfd6ae1ea130e445`
- 模型哈希：`sha256:22da89ea64b2bdfd105882462a9565549b8f7bcfe4ac13ab837ad5529f78c42a`

### DAgger-v2

`/code/rosclaw/rosclaw_football/evidence/s57-corrective-student-dagger-v2/normal-dagger-round2-maxchannel16-gain016-seed5600/student-report.json`

- 报告哈希：`sha256:1bd6cdc150432cb728b7057416dee486b097e75d2a39942036fb635387140eb8`
- 模型哈希：`sha256:8e9ab132ec5455a423f9ff2733f54b3c9169fe5c1dac69b84abd2140e852a240`

### confidence-gate-v1

`/code/rosclaw/rosclaw_football/evidence/s57-corrective-student-gate-v1/plastic-v1-silence-v2-logistic-ood999-seed5600/student-report.json`

- 报告哈希：`sha256:cbb07f69985075c6e32254252cc9dcdac149463ff52513ed8548912409603026`
- 模型哈希：`sha256:319a0ffa73de3b8bdccf2f2ef22ae343e8b6c0430c74f89ac53df5449998a959`

### gate-v2

`/code/rosclaw/rosclaw_football/evidence/s57-corrective-student-gate-v2/yaw-conflict-pruned14-gain022-seed5600/student-report.json`

- 报告哈希：`sha256:f7976b2d4c431bc41c9f80e916c347a2305e225a374798823c7bfb419ab47863`
- 模型哈希：`sha256:d9496d4dfff9d62fa13bd6cc4d392a7feb21cef3f2ebea44f2ee51efddba9909`

四份报告均可由当前验证器重新加载；权限上限均为 `SIM_ONLY`，`deployment_candidate=false`、`promotion_eligible=false`、`promotion_authority=NONE`、`hardware_authorized=false`。

## 软件验证

- `ruff check src tests`：通过；
- `mypy src`：170 个源模块通过；
- `pytest -q`：`421 passed, 96 skipped`；
- 本轮 4 个触及的 Python 文件 `ruff format --check`：通过；
- 仓库全量格式检查仍报告 49 个既有未格式化文件，本轮没有机械改写这些用户存量变更。

96 个 skip 主要来自仓库基础环境未安装 torch/JAX 的既有可选测试以及本地未具备资格的外部资产；S57 本身没有以 import skip 代替物理验证，而是在 OpenTrack 隔离环境中实际使用 4 张 A6000 完成 MJX 状态推进、训练、父/子对照和外部证据写入。

## 代码入口

- `src/rosclaw_soccer/training/recovery_corrective_student.py`
  - DAgger 正常回放混合；
  - 独立稳定门；
  - 稀疏通道“最多 N 个净有益通道”契约；
  - 潜变量逻辑置信门、OOD 包络、NumPy 推理与证据验证。
- `src/rosclaw_soccer/training/opentrack_recovery_corrective_student.py`
  - 四卡候选正常轨迹采集；
  - 多轮 DAgger；
  - 置信门训练与 MJX 双考；
  - 旧候选的哈希绑定保守复考。
- `tests/test_s56_recovery_corrective_student.py`
  - 平衡且源对齐的 DAgger 混合；
  - 独立稳定门；
  - OOD 零输出；
  - 证据/权限/模型/语料篡改拒绝。

## 下一步：S58 应做什么

1. **时序置信门**：当前逐帧逻辑门不能表达“刚开始救援后应保持一小段时间”或“退出后暂不重新触发”。下一步使用小型 GRU 或显式滞回状态，训练开门、保持、退场三类标签，并单独约束 gate slew。
2. **冻结当前 24 个留出源**：不再围绕同一批状态调锐度或增益。扩展到 `40/80/160/400+` 新失败状态和新的正常路线种子，当前留出只作为回归集。
3. **序列级反事实教师**：局部首状态 Jacobian 无法预测 600 步闭环干扰。应在短序列上估计通道对稳定、偏航和横向的累积因果作用，再做组稀疏选择。
4. **门控 DAgger**：采集置信门自身造成的开/关边界状态，而不是只采未门控学生；把误开和过早关闭分别标注。
5. **阶段制品化**：将 96×600 DAgger 轨迹单独原子落盘并绑定源报告哈希，使门/通道试验无需重复昂贵物理采集。
6. **编译核优化**：在逐步版本作为数值真值的前提下，实现 20 步可复用块并做逐位一致性检查，降低长考主机调度成本。
7. 只有新候选同时通过未见失败、正常总代价、方向、独立稳定和有限性五个门，才进入 CPU MuJoCo 真值复考和足球恢复视频链。

## 不能声称什么

- 没有训练出可部署小脑；
- 没有证明真实机器人安全或授权；
- 没有通过 CPU MuJoCo 晋升考场；
- 没有跨动作、跨数据集或跨任务泛化结论；
- 本轮没有合格宣传视频。视频必须服从物理证据，不能反过来替候选背书。
