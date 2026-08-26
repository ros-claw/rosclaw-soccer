# S9 Goalkeeper V2：在线 PPO 第一轮闭环实施报告

日期：2026-08-13
边界：`SIM_ONLY`；没有打开 ROS、DDS、串口、CAN 或厂商 SDK；没有向真实机器人发送命令。
物理裁判：CPU MuJoCo 数值轨迹；视频不参与晋级判定。

## 1. 本轮结论

本轮已经真正打通以下闭环，而不是只增加一段门将补丁：

```text
MotionDecode 动作数据
  → 条件化门将动作库
  → 4×A6000 在线 PPO / 非对称 Actor-Critic
  → JSON 数值 Actor Candidate
  → 冻结父策略 + 同球同 seed 的 CPU MuJoCo 考试
  → fail-closed 晋级门
  → 失败场景诊断
  → 缩小危险可塑边界并重新训练
  → CPU MuJoCo 再考试
```

出现了一个重要但有限的突破：候选的测得反应延迟由父策略的 80 ms 降到 0 ms；第一代学习策略的安全失败为 7/25，经过失败回灌和受限可塑边界后降到 0/25，并在两次严格重放中逐项一致。

但是候选没有提高整体扑救覆盖、触球率或真扑救率，所以晋级门正确给出 `REJECTED`。当前 Champion/父策略没有被替换。

这说明 Growth 闭环已经能识别并修复“学得更快却更容易摔倒”的伪进步，但还没有训练出覆盖更大的全身门将。

## 2. 新增的工程能力

### 2.1 MotionDecode 门将动作库

新增 `skills/goalkeeper_v2/motion_library.py`，从本地完整 MotionDecode 数据生成内容寻址、许可绑定的动作库：

- READY
- SPLIT_STEP
- SHUFFLE_LEFT / RIGHT
- LOW_SAVE_LEFT / RIGHT
- HIGH_REACH_LEFT / RIGHT
- CENTER_BLOCK
- RECOVERY

当前所选片段是门将动作的训练代理，不是人体门将标注。Manifest 明确记录 `proxy_only=true`，禁止把它宣传为真实 goalkeeper demonstration。左右动作通过显式镜像生成；每个源文件、MotionDecode README/License、29 关节顺序和 G1 Body hash 都进入内容哈希。

动作库：

```text
/code/rosclaw/rosclaw_football/evidence/s9-goalkeeper-motion-library-v1/
library hash: sha256:7ece0599df369935fa11b13de868b5410b6bb22351a4e613cda334b2fcdf5dc9
```

MotionDecode 许可被按本地 `LICENSE.md` 约束为非商业研究/个人学习/原型训练用途，数据与训练产物均留在 Git 外。

### 2.2 端到端因果 Actor

新增 JSON 数值 Actor：

```text
Actor input (120):
  最近 8 帧球的相对位置
  + 重力方向
  + 根部线速度/角速度
  + 29 关节 q/dq
  + 上一帧学习残差

Actor output (30):
  1 个侧向速度
  + 29 个有界全身关节 position residual
```

部署 Actor 不能看到射手内部 phase、未来轨迹或 Critic 特权信息。Actor 使用 JSON 数值序列化，不使用 pickle；加载时校验内容哈希、Body hash、观察契约 hash 和 Parent hash。

### 2.3 在线 PPO 与非对称 Actor-Critic

新增 `training/goalkeeper_ppo.py`：

- 4 卡 DistributedDataParallel 在线 PPO；
- Actor 仅使用 120 维因果观测；
- Critic 可额外使用真实球速、拦截点、区域、接触和手位置；
- 目标 episode 契约为至少 3 秒，且扑到球或接触球不得终止；
- 当前 bootstrap 按 READY / FLIGHT / LANDING / RECOVERY 做独立 phase-conditioned reset 采样；
- MotionDecode 仅作为 train-only condition/style regularizer；
- GPU 快速分析世界只能生成 Candidate，不能生成 Champion；
- 训练结束校验四个 rank 的参数最大差异，非同步即失败。

当前 fast world 是向量化、单步的分析训练世界，不是 Isaac/MJX 高保真多步动力学；训练报告明确记录 `multi_step_episode_training=false` 和 `continuous_reset_enabled=false`。它足以验证在线策略更新、四卡同步、阶段课程和候选导出闭环，但还不能声称完成了 3 秒连续 episode 训练，也无法替代 CPU MuJoCo 的稳定性与接触考试。

### 2.4 匹配式严格考试与晋级门

新增/加强：

- 五档 Coverage-Time：1.0 / 0.8 / 0.6 / 0.5 / 0.4 秒；
- 每档 upper-left / upper-right / lower-left / lower-right / center 五球；
- 父策略与候选使用相同 launcher、相同目标、相同 seed；
- scenario suite hash 只绑定考试题，不再错误绑定策略结果；
- 每球记录明确的安全失败码和最小骨盆高度；
- 父子各跑两遍，逐项验证 strict replay；
- 晋级门缺证据即拒绝。

硬门包括：反应 p50/p90、固定时间覆盖、触球、真扑救、恢复、二次扑救、人体先验、安全、历史技能、严格重放和密封留出集。

## 3. 四卡训练结果

最终本轮使用的第三代训练任务：

```text
/code/rosclaw/rosclaw_football/evidence/s9-goalkeeper-ppo-4gpu-dev-v7/
```

| 指标 | 结果 |
|---|---:|
| GPU | 4 × NVIDIA RTX A6000 |
| 并行训练样本 | 327,680 |
| 初始平均代理奖励 | 0.395644 |
| 最终平均代理奖励 | 0.533126 |
| 最优平均代理奖励 | 0.535035 |
| 最终 save proxy | 0.477501 |
| 最终 recovery proxy | 0.840970 |
| 四卡参数最大差异 | 0.0 |
| Candidate policy | `sha256:2f62a515750f946699d721f4805f0d78d8b51934b79c293cce62a8dbf3caa39a` |

训练代理奖励确实上涨，但该数字不能被解释为 MuJoCo 扑救成功率。

## 4. CPU MuJoCo A/B 结果

正式证据：

```text
/code/rosclaw/rosclaw_football/evidence/s9-goalkeeper-ppo-4gpu-dev-v7-cpu-exam-v1/
evidence hash: sha256:aefa6bdc346fb09efa424f394f5032425bebffa5f0a8b53581fbf5d69715e213
decision hash: sha256:c4a7ca987ad6ae6d488788dc08771f7b9540b70adc9ba9a9563bfadcdad7cbec
verdict: REJECTED
```

父策略与最终候选的 Coverage-Time 对比：

| Deadline | Parent coverage/contact/save | Candidate coverage/contact/save | Parent → Candidate reaction p50 | Candidate safety |
|---:|---:|---:|---:|---:|
| 1.0 s | 20% / 20% / 20% | 20% / 20% / 20% | 80 ms → 0 ms | 0 |
| 0.8 s | 20% / 20% / 20% | 20% / 20% / 20% | 80 ms → 0 ms | 0 |
| 0.6 s | 0% / 0% / 0% | 0% / 0% / 0% | 80 ms → 0 ms | 0 |
| 0.5 s | 0% / 0% / 0% | 0% / 0% / 0% | 80 ms → 0 ms | 0 |
| 0.4 s | 0% / 0% / 0% | 0% / 0% / 0% | 80 ms → 0 ms | 0 |

两次 parent replay 和两次 candidate replay 均完全一致。

通过的硬门：

- median reaction latency；
- p90 reaction latency；
- safety regression；
- strict replay。

未通过或仍缺证据：

- fixed-deadline coverage；
- save contact rate；
- true save rate；
- recovery time；
- second-save success；
- human-motion prior score；
- historical keeper regression；
- sealed holdout。

## 5. 失败回灌实际修复了什么

### 5.1 第一代全身 Candidate

第一代正式候选出现 7/25 安全失败：

- 7 次骨盆低于 0.60 m；
- 其中 3 次伴随全局/门将关节越界；
- 失败集中在 upper-left、left 和 center 的部分课程；
- 最低骨盆高度仅约 0.067 m。

它反应快，但会倒，不能晋级。

### 5.2 受限可塑边界

将 29 关节学习残差幅度变成训练配置的一部分并进入候选哈希，默认缩放为原上限的 25%。结果：

```text
safety failures: 7/25 → 0/25
reaction p50: 80 ms → 0 ms（优势保留）
```

这不是把策略“退回手写动作”：侧向速度和 29 维残差仍由神经 Actor 输出；只是 ROSClaw 的安全边界不允许新手一次获得过大的肌肉权限。

### 5.3 修复训练/部署契约错位

发现并修复两个关键错误：

1. 腰部目标输出索引偏移一维；
2. 训练的 `previous_action` 是上一帧学习残差，部署却误喂了 locomotion 的绝对关节目标。

同时取消“学习残差 + 旧手写扑救 reach”叠加，CPU 考试现在测的是学习 Actor 自己的全身 reach，而不是两个控制器混在一起。

### 5.4 修复任务奖励被 30 维稀释

原 save proxy 对 30 维动作直接取均方误差，真正决定覆盖的侧向动作只占 1/30。策略可以靠改善较小的姿态输出提高训练分，却没有足够动力快速横移。

现改为：

```text
75% lateral intercept objective
+ 25% whole-body posture objective
+ train-only task-action auxiliary
```

训练代理 save proxy 从上一代约 0.423 上升到约 0.478，但 CPU 覆盖仍未增长，说明下一瓶颈不是继续调奖励权重，而是 fast world 缺少真实 locomotion dynamics 与接触/稳定性反馈。

## 6. 为什么 0 ms 不是“超人反应”

这里的 0 ms 是离散控制指标：观察器第一次由可见球历史确认来球的同一控制周期，Actor 就输出了超过阈值的有效动作。因此在 20 ms 控制周期分辨率下记录为 0 ms。

它不代表真实相机、网络、推理和执行器总时延为零，也不授权真实机器人。下一步应把感知/执行延迟随机化加入 GPU 训练和密封留出考试。

## 7. 通俗解释

可以把这轮看作培养一个守门员学员：

1. 先给他看一组站立、侧移、单脚支撑、接球和恢复动作，形成动作词典；
2. 四张 GPU 同时让很多虚拟学员在线试错；
3. 学到的不是一段固定脚本，而是“看到最近几帧球和自己的身体后，下一步怎么横移、29 个关节各动多少”的小脑网络；
4. GPU 训练场只负责练功，正式考试仍在更严格的 MuJoCo 里逐球打；
5. 第一名学员反应非常快，但扑左边时经常把自己扑倒；
6. ROSClaw 没有因为训练分高就放行，而是记住这些失败、缩小新手的动作权限、重新训练；
7. 第二/三代不再摔倒，反应仍然快，但还没有扑到更多角落球，因此继续留在 Candidate 班，不能替换老门将。

这正是自进化与普通调参的区别：系统不只会“生成一个更高分模型”，还必须知道自己在哪个世界变好、在哪个世界失败、失败是否可复现，以及有没有资格替换现役能力。

## 8. 验证

```text
pytest: 190 passed, 4 skipped
mypy: Success, 81 source files
ruff check: passed
compileall: passed
4-GPU DDP parameter maximum difference: 0.0
CPU MuJoCo strict parent replay: true
CPU MuJoCo strict candidate replay: true
```

4 个 skip 是已有的堆叠 Core PR / 旧实现迁移条件，不是本轮 Goalkeeper V2 失败。

## 9. 下一轮精确任务

下一步不应盲目继续增加 PPO 轮数，优先级如下：

1. 把快速分析 world 替换/增强为 GPU 刚体动力学 world（Isaac Lab、MJX/MJWarp 中择一），保留 CPU MuJoCo 作为唯一晋级裁判；
2. 把这轮 7 个跌倒场景与未覆盖 upper/lower corner 场景生成内容寻址 nightmare curriculum；
3. 在真实动力学训练中加入骨盆高度、支撑多边形、角动量、关节界限、力矩和接触奖励；
4. 增加 3 秒连续双球回合，实测 recovery time 与 second-save rate；
5. 训练真正的 position-conditioned AMP/discriminator，并产生非空 Human Motion Score；
6. 引入 actor delay/randomization 和球速、摩擦、质量、PD 的 domain randomization；
7. 建立独立历史门将池和 ≤3% regression exam；
8. 所有开发集指标满足后，才打开一次密封 holdout；失败后重新封存新 holdout，不能反复刷题。

本轮没有制作“成功宣传片”，因为候选未晋级。下一段宣传视频应该建立在真正提高 Coverage-Time、出现侧扑/高扑并能恢复二次扑救之后；开发视频必须醒目标记 `REJECTED CANDIDATE / NOT PROMOTED`。

本轮已输出一段 12 秒、720p、30 fps 的诚实开发对比视频（upper-left challenge 与 center matched save，父策略/候选各自完整数值轨迹）：

```text
/code/rosclaw/rosclaw_football/evidence/s9-goalkeeper-ppo-4gpu-dev-v7-cpu-exam-v1/goalkeeper-v2-rejected-development.mp4
video hash: sha256:2daeb9c6ccae5aa875ae1698765c09338cdf5a0f9119275e681233a7167fca6c
```

视频显式标注 `REJECTED CANDIDATE · NOT PROMOTED · CPU MUJOCO · SIM ONLY`，只用于观察动作，不参与打分。
