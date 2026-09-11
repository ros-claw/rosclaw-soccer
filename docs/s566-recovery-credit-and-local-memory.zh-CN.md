# S566–S573：恢复奖励归因与局部动作记忆

这是研发记录，不是宣传验收报告。所有实验仅限 SIM_ONLY，没有真实硬件动作，也没有新的整队宣传视频。

## 问题与实现

此前 clone1000 在单个接球身体上下文的小扰动测试中出现成功转传，但普通 PPO 和行为克隆并不保证继续进步。此次保留失败候选，不把训练损失下降当作球技提升。

新增两个独立训练模块，不引用具体 G1 关节名称、足球场对象或执行器：

- `training/fixed_recovery_credit.py`：对采集器明确保证“不再使用新策略输出”的固定恢复后缀，将实际后缀折扣回报折入最后一个真正执行的策略动作。PPO 仅学习前缀；验证终止掩码、死后奖励、非有限值与回报溢出，不修改原始证据。
- `training/local_action_memory.py`：按观测中的相位以及当前特征检索成功经验。经验中心与原动作是冻结缓冲区；仅允许学习有界的动作修正。相位、维度、有限性、归一化和修正幅度均受校验。它是非参数动作记忆加可训练修正，不是已经训练成功的端到端力矩策略，也不授予动作权限。

它们暂在 Soccer 训练包中孵化，尚未迁移或声称成为 Core 的正式通用产品能力。

## 统一物理判据

沿用原 CPU MuJoCo 3.12、50 Hz 控制、每控制帧 10 个 0.002 秒物理步；每个世界 210 控制帧。前 50 帧策略输出进入原残差限制，之后使用明确的零导航和残差衰减恢复。恢复不是新学出的神经网络。

转传成功仍要求实际脚接触触发有效定向出球、球到达目标 0.3 米以内、低球高度、全程身体安全、无非脚触球、不出界，以及原来球速度和飞行期限条件。没有放宽到达阈值或碰撞条件。

目标是历史真实队友位置，但隔离学习世界没有物理队友接球，其他球员的上下文来自记录。所有统计仍围绕一个已消费的身体上下文，不能称为完整传接射门或 4v4 成功。

## 已完成实验

| 实验 | 训练与检查 | 严格转传结果 |
| --- | --- | --- |
| S566 | clone1000 起点；16 轮真实 PPO；每轮 32 个新扰动；128 次优化器更新、23,227 个有效策略前缀样本 | 开发集起点 11/32，最终 8/32，没有优于起点；512 个训练回合中 57 个成功 |
| S568 | 同种子奖励对照：完整成功奖励 30→100，方向/距离过程奖励乘 0.1 | 开发集起点 11/32，最终 8/32；没有提升，不替换旧策略 |
| S567 | 从上述 57 个成功训练回合提取 2,850 个前缀动作样本；加 0.25 系数旧策略锚定；actor 学习率 1e-4 | 开发集最佳 14/32，原策略 12/32；冻结候选换到新样本后 9/32，配对原策略 11/32，拒绝晋升 |
| S570 | 同一语料的 1e-3 学习率对照 | 起点 12/32，后续最高仍为 12/32；最终 4/32，拒绝晋升 |
| S571 | 成功动作记忆，逐控制帧按实际观测检索，不重放固定整条动作序列 | 邻居数 1/4/8/16/32 对应 18/19/18/12/23 个成功；同开发集原神经策略 12/32 |
| S572 小扰动 | 预选 32 邻居，换到配对新种子 56901 | 记忆 21/32，原神经策略 11/32；双方身体安全 32/32 |
| S572 四倍扰动 | 同种子、扰动幅度四倍 | 记忆 5/32，原神经策略 4/32；仍不稳健，不能宣传为通用突破 |
| S573 | 将原型替换为可复用的 `local_action_memory`，修正初始为零 | 开发集 23/32；与 S571 对应测试完整 qpos/qvel 最大误差均为 0 |

小扰动分别为球 XY ±0.005 米、球 XY 速度 ±0.01 米/秒、29 个关节初值 ±0.0005 弧度。四倍扰动为 ±0.02 米、±0.04 米/秒和 ±0.002 弧度。不能将这两个分布的成功率混合汇报，也不能将重复开发集筛选当作盲测。

## 验证与证据

S566 首次更新从安全加载的 checkpoint、完整 rollout、折叠后的前缀和 RNG 独立复算，所有参数最大误差为 0。冻结底层运动策略未被优化器更新。

新增模块相关检查：86 项聚焦测试通过；两个新模块 mypy 通过；Ruff 和 compileall 通过。还显式配置真实本地 G1 资产，补跑共享世界、4v4、守门相关 45 项测试，全部通过。测试通过不等于比赛效果验收通过。

新增动作记忆模块之前的全仓库检查为 1,910 passed、29 skipped、11 failed。11 项均为旧外部证据绑定校验失败，在未修改的 27c5b72 临时基线工作树中原样复现（相关文件 48 passed、3 skipped、同样 11 failed）。没有删除旧证据、修改哈希或绕过这些失败。新模块之后仍需更新全量检查记录。

完整轨迹、NPZ、checkpoint 和实验脚本留在本地外部证据区，不进入 Git。主要目录：

- `/home/dell/rosclaw_soccer_evidence/s566-recovery-credit-ppo-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s567-success-consolidation-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s568-outcome-credit-ppo-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s569-frozen-baseline-narrow-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s569-frozen-baseline-wide-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s569-frozen-consolidated-narrow-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s570-success-consolidation-strong-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s571-success-memory-exam-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s572-success-memory-fresh-scale1-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s572-success-memory-fresh-scale4-v1/train`
- `/home/dell/rosclaw_soccer_evidence/s573-modular-memory-exam-v1/train`

脚本位于 `/code/rosclaw/phase8_evidence/s532-delayed-handoff-cpu-v1/`。本次没有录制新的交付视频，避免将局部成功剪辑误作用户要求的整体比赛宣传效果。

## 继续实施的边界

S574 正在使用冻结经验中心、有限可塑修正及 actor-critic，对四倍扰动继续训练。首次更新已独立复算，参数与缓冲区最大误差为 0；最终效果尚未验收，不能提前写成成功。

优先补足困难上下文的成功经验，再做旧能力保留检查，最后回接实际多身体世界的合法角色目标、队友接球与后继动作。单个身体的到点事件不能替代真实队友的接球，也不能替代球队协作验收。
