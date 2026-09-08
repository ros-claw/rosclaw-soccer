# S227：检查“球留在脚边”的奖励偏置

## 动机和证据边界

S224 小批续训和 S226 平衡大批续训均未改善整体足球表现。审阅奖励实现发现，
旧接近球项是 `Phi(after) - Phi(before)`，而 GAE 使用 `gamma=0.997`，且回合
结束时没有消去终态势函数。它不满足本有限课程的折扣势函数构造，可能把
结束时离球较近当作额外收益；这不等于已经证明它是所有失败的根因。

研究依据：Ng、Harada、Russell，ICML 1999，*Policy invariance under reward
transformations: Theory and application to reward shaping*，第 3 节公式 (2)。
作者公开 PDF 已保存为证据目录的 `reference-shaping-icml99.pdf`。

## 独立、可回退的新奖励合约

`--reward-shaping terminal_potential_v1` 保留所有真实触球、成功接力、非脚碰球、
姿态、碰撞和能耗项，仅替换接近球项：

```text
Phi_t = 2 * exp(-4 * min(left_ankle_distance_t, right_ankle_distance_t))
F_t = gamma * Phi_(t+1) - Phi_t
Phi_terminal = 0
sum(gamma^t * F_t) = -Phi_initial
```

使用每帧动作前的原始观测计算势函数，相邻帧按真实顺序相接。末帧进入
本课程的有限回合终态，不借用下一回合的观测。单元测试覆盖 1/3/600 帧、
三个折扣、改变末帧球位、非有限值和非法形状。另测试真实接力奖金和
非脚中断规则在新旧塑形模式下都保持原定义。

`legacy` 默认仍可复算旧实验；新模式和 gamma 写入训练 manifest 与每名角色
的 Core 更新上下文，优化器重算按原模式执行。没有修改球的位置、质量、
碰撞体或控制力矩限制。也没有因读到一篇论文就宣称 PPO 保证收敛：这是
有限轨迹的代数校验，不是部分可观测、多智能体神经 PPO 的最优性证明。

## 固定对照协议

与 S226 相同的第 48 代父模型、8 worker、五球位平衡批次、两次更新和
相同种子；只改变塑形项。首批 40 场必须与 S226 物理轨迹完全相同，之后
允许因学习权重不同而产生差异。父子各 16 场开发验收，继续采用原阈值。
旧 critic 未重置，需适应新的回报目标，这也是本次短训练的局限。

证据目录：`$ROSCLAW_SOCCER_EVIDENCE/s227-terminal-potential-private-ppo-v1/`。
当前状态：实验运行中，尚无成功率改善结论；候选不自动晋升。

首批对照已核验：40 场主轨迹及独立重放与 S226 逐一一致；只有更新后的
权重不同。S226 第 49 代为 `sha256:30cbc6634d49daa4dd68335b4028a932255d8adaf5d0a538c807cbf7e7ab858a`，
本轮为 `sha256:f756f012ff42f64b9c2296cb815b7dd57ea923dd29d5b263dff343d341d7113c`。
因此两者的第一步梯度对照没有混入不同物理采样。

回归：1171 passed、13 skipped、11 个既有外部历史证据绑定失败；ruff、mypy
（405 个源文件）及 compileall 通过。S219 的旧双场更新也通过新版重算器
精确复算，兼容性报告单独保存，没有覆盖旧证据。

本阶段仍是冻结步行基础上的腿部近球残差学习；不训练射门或扑救专用
上肢策略，也不等于已经接通完整比赛的所有专用运动技能。
