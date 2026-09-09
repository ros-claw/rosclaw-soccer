# S350–S351：把真实小脑记忆接入批量训练，而非只拷贝机器人姿态

结论：完成了真实六人状态到 CPU/GPU 基础行走推理的可校验桥梁。
这是训练基础设施进步，不是新的合格比赛或已经提升的停球技能。

## 发现：小脑还有 256 维 LSTM 双状态

实际 RoboNaldo `policy_29dof.pt` 包含 LSTM，隐藏态与记忆态各为
`[1,1,256]`，并保存归一化统计。上一帧 action 只是输入的一部分。
S347 修复的是 action 的坐标表达，**没有解决 LSTM 隐状态的换侧语义**。
后续不能把“补了 action”说成“完整小脑记忆已恢复”，也不能随便镜像隐藏态。

被动记录实际八人基础策略推理的观察、上一帧 action、输出目标、增益及
镜像模式；原有六人的控制前 qpos/qvel 同时保存。基础策略可能是被全身
模型覆盖的影子输出，文件明确标注 `FOUNDATION_INFERENCE_SHADOW_NOT_APPLIED`。
主物理轨迹与旧候选逐数组完全一致，记录没有改机器人或球的动作。

## 没有通过放宽数值误差掩盖重放问题

初版只恢复 action，批量调用失败：模型内部 LSTM 的 batch 仍为 1。
补循环态后，直接改成 eval/no-grad 的序列批量推理也超过初定误差界限。
检查发现原导出模型保留 training/gradient 前向路径，其构造阶段先执行
50 次零输入 inference-mode warmup。只改前向模式会改变数值路径。

按真实构造方式重放，保留原前向路径，只在每步后 detach 循环状态：
六名场上球员各 1000 帧，观察、action、关节目标均 **零误差**。
由此重建完整循环态快照，未将缺失状态假填为零。

用这些完整快照做 6000 个独立单步批量推理：

| 后端 | 最大观察误差 | 最大关节目标误差 |
|---|---:|---:|
| CPU，Torch 2.13 | 0 | 1.431×10⁻⁶ rad |
| A6000，Torch 2.11 CUDA | 0 | 9.537×10⁻⁷ rad |

均通过原定 5×10⁻⁶ rad 单步目标界限。没有把单步推理一致性称为闭环物理
一致性；CPU/GPU 物理迁移和整场保留还须另外考试。

## 循环梯度图寿命：值不变，但本轮没有整体内存收益

原基础策略推理参数仍带梯度标志，LSTM 状态会持有 `CopySlices` 图。
新增 `detach_frozen_recurrent_state`，只用于不属于任何优化器的冻结推理
模型：验证状态对布局/有限值，保留数值、dtype、设备、存储，切断梯度链。
它不是训练器，也不是动作入口，不重置记忆或改变模型权重。

S351 红蓝 × detach 开/关四条 20 秒八体课程：完整轨迹、八人的循环态和
观察快照都精确相等，身体安全、零机器人碰撞。峰值 RSS 均约 10.6 GiB，
耗时约 205–208 秒，**没有测出明显整体内存或速度改进**。模型/session
常驻可能占主要开销，但这里只把它作为解释假设，不宣称已定位全部占用。

## 下一步

使用真实接球过程中的 qpos/qvel、上一帧 action 和完整 LSTM 双状态建立
隔离接球课程，训练新的 29 关节残差小脑。它是新的收球技能，不替换旧
SONIC 射门模型；必须回到 4v4 验证。训练电机上限显式降低至与共享世界
一致的 85%，不能让 GPU 练强力电机、再声称能直接迁移。

证据目录：

- `/code/rosclaw/phase8_evidence/s350-measured-locomotion-training-bridge-v1`
  中 `scalar-parity.json`、`reconstructed-lstm-scalar.npz`、
  `snapshot-batch-cpu.json`、`snapshot-batch-cuda-0.json`。
- `/code/rosclaw/phase8_evidence/s351-frozen-lstm-graph-lifetime-v1`
  中 `assessment.json`、四条原始轨迹及完整循环态快照。

所有阶段 SIM_ONLY，无真实硬件操作，无整场比赛晋升。
