# S5：MotionDecode 全身模仿与稳定可塑闭环

日期：2026-08-12
边界：`SIM_ONLY`、CPU MuJoCo、无 ROS/DDS/厂商 SDK/硬件命令

## 结论先行

本轮不再把“自然”当作视频观感，也没有把 MotionDecode 原始 CSV 直接当成力矩策略。我们把已经炼化并哈希绑定的 MotionDecode v6 全身姿态/速度先验接入同一个三智能体、单足球的 MuJoCo 闭环，再让 Growth 在安全、精度、模仿、平滑、后退和滑移共同约束下选候选。

最终候选通过开发门，但仍明确为 `PASSED_DEVELOPMENT_GATE_NOT_PROMOTED`：

| 指标 | 父技能 | 模仿候选 | 变化 |
|---|---:|---:|---:|
| 6.64 m 射门落点误差 | 0.0551 m | 0.0177 m | -67.9% |
| 踢后峰值后向速度 | 0.0289 m/s | 0.00174 m/s | -94.0% |
| 踢后支撑脚滑移 | 0.0736 m | 0.0568 m | -22.8% |
| 躯干最大侧倾 | 0.2609 rad | 0.2306 rad | -11.6% |
| 触球窗关节加速度 RMS | 93.37 rad/s² | 91.55 rad/s² | -1.9% |
| 踢后关节加速度 RMS | 51.45 rad/s² | 49.90 rad/s² | -3.0% |
| 踢后根部加速度 RMS | 2.913 m/s² | 2.781 m/s² | -4.5% |

通过的物理事实还包括：9.04 m/s 射门、2.33 cm 传球到点误差、无跌倒、无关节/力矩/执行器违规、真实滚动审计通过、双重回放逐项与轨迹摘要完全一致。

## 这次到底学了什么

可以把控制器理解为三层：

1. 原来的足球技能负责“球怎么踢准”；
2. MotionDecode 教师负责“腰、双臂、支撑腿和踢球腿应该怎样协同经过动作”；
3. 小脑恢复/步态策略负责“触球后把冲量带走，继续稳定地走出来”。

本轮补上的关键缺口是第二项的“速度教师”。仓库以前已有 MotionDecode v6 速度参考和有界融合函数，但三智能体真实 PD 力矩仍固定使用零目标速度，等于自然摆动被阻尼项压掉。现在力矩仍只由审计过的 PD 链生成：

`tau = kp * (q_target - q) + kd * (dq_target - dq)`

模仿只提供小幅目标残差：姿态峰值 0.006 rad、速度峰值 0.08 rad/s；随后仍经过关节预测保护、85% 力矩权限投影和硬力矩上限。数据教师没有直接输出力矩，也不能绕过安全链。

## 稳定性—可塑性如何处理

“模仿越多越好”是错误命题。足球碰撞会把脚面几毫米/几毫弧度的差异放大为明显落点偏差。粗搜索中出现过：动作加速度和滑移改善，但落点误差扩大到 0.19–0.23 m。它们均被精度记忆门拒绝。

本轮把选择条件写成机器可执行的联合门：

- 父技能精度不得遗忘；
- MotionDecode 姿态和速度跟踪误差都不得变差；
- 触球与踢后关节加速度不得变差；
- 踢后根部加速度、后退速度、支撑脚滑移、躯干侧倾不得变差；
- 尾段晃动只允许 10% 的窄容差；
- 任一 NaN、跌倒、关节越界、力矩越界或执行器饱和立即拒绝。

最终搜索的 3 个精调候选全部安全通过，但 0.06 m/s 的触球后自然随动速度综合分最高。更早的 2 个粗候选分别因精度遗忘、教师误差或后退退化被拒绝。这就是“成功、失败都进入 Growth 判断”，而不是人工看完视频挑一个。

## ROSClaw 模块闭环

- `growth/football_motion_prior.py`：加载并验证 MotionDecode v6 哈希绑定的全身姿态/速度教师；
- `skills/team/shared_world.py`：三个 G1 和一颗真实物理足球共享同一 MuJoCo 时间线；教师通过 PD、关节保护和力矩权限链执行；
- `skills/team/imitation_learning.py`：候选课程、自然度量、稳定可塑门、自动选择；
- `skills/team/imitation_evidence.py`：父技能、候选搜索、双重严格复演、滚动审计和证据固化；
- `evidence/three_player.py`：重新计算实现哈希、请求/轨迹承诺和物理指标，防止视频替代数值证据；
- `media/three_player_video.py`：只消费通过验证的证据，像素不参与评分；
- ROSClaw 扩展 CLI：`rosclaw soccer academy train-imitation`。

这实现的是可复用的“教师先验 → 安全残差 → 多目标反事实 → 失败拒绝 → 严格复演”模式。虽然 G1/MotionDecode 解释仍属于 Soccer 下游，Growth 的证据、门控、不可越权和不晋升语义仍由 ROSClaw 通用原则约束。

## 证据与视频

- 证据：`/code/rosclaw/rosclaw_football/evidence/s5-imitation-growth-v2/g1-imitation-growth.json`
- 轨迹：`/code/rosclaw/rosclaw_football/evidence/s5-imitation-growth-v2/trajectory.npz`
- 1080p 视频：`/code/rosclaw/rosclaw_football/evidence/s5-imitation-growth-video-v2/g1-motiondecode-imitation-growth-1080p.mp4`
- 视频时长：36.73 s，1920×1080，30 fps，1102 帧，H.264
- 证据 SHA-256：`73d0bfc0758536b9b2866c89dc8510462fbecb92cff920a0f3616ea20f4e07bf`
- 轨迹摘要：`c7d896d64e53b810c7123e4b304a58ff639995428ec59fcf2ce2da78a93acf08`
- 视频 SHA-256：`69f7335908c42720a88943c62d2ed2c2b979270c606c78159cbedb35a4a02a8b`

## 未完成与下一轮

这不是端到端自然人动作的终点。当前先验只在触球附近约 1 秒工作；跑动接近、刹停、摆腿、随动和恢复仍由多个专家串联。下一轮应把 OmniContact 的球—脚接触示范用于接触相位教师，把 MOSAIC/MotionDecode 的跑—停—踢转换用于更长时域的相位条件模仿，再用当前硬门训练可持续更新的神经残差 actor。任何更强教师都必须保留本轮的精度记忆、后退和安全反事实门。
