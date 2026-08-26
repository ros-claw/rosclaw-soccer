# S106：上下文门控的神经恢复策略

## 结论

S106 修复了 S105 暴露出的两个具体问题：

1. `left-outer` 父策略本来不需要横向恢复，但神经网络仍产生了很小的近零残差；
2. 留出的 `right-inner` 虽然动作峰值明显更平滑，但总变差略高于父策略。

本轮没有重新训练，也没有降低考试阈值。我们在神经 actor 与冻结 locomotion policy 之间加入一个因果、单调、稀疏的 authority envelope：actor 仍负责提出命令幅值，envelope 只允许命令减小或归零，不能扩大任何一个输出分量。

正式结果：四条球路的完整传球、射门、高空手套扑救、落地、恢复、二次横移和再次 ready 全部通过；候选严格复放通过；组合横向命令总变差比父策略下降 `33.04%`；四条球路的峰值命令步长全部不劣于父策略；零动作球路恢复为精确零；没有任何 ready 时间退化。

## 1. S105 的失败归因

S105 已经在组合指标上取得 `29.53%` 的命令总变差下降，但逐球路并非全优：

| 球路 | S104 TV | S105 TV | 现象 |
|---|---:|---:|---|
| left-inner | 0.413337 | 0.226136 | 明显改善 |
| left-outer | 0.000000 | 0.030620 | 零动作上下文出现网络残差 |
| right-inner（训练留出） | 0.126201 | 0.147243 | 总变差略增，但峰值步长大幅下降 |
| right-outer | 0.306072 | 0.191918 | 明显改善 |

离线轨迹诊断进一步发现：

- `left-outer` 的 pelvis 横向位置始终在 `±0.15 m` 合格死区内，709 个 actor 激活帧却都有非零横向残差；
- `right-inner` 并非高频跳变。其神经命令峰值步长约 `0.0054 m/s`，父策略为 `0.0914 m/s`；TV 稍高来自连续小幅调整，而不是一次大跳；
- 部分状态上网络会提出与误差方向相反的极小命令，虽然不足以破坏稳定，但没有保留价值。

因此问题不是“RL 训练量不够”，而是函数逼近模型在零附近天然很难保证精确稀疏，也缺少解析的单调权限边界。

## 2. Authority envelope

每一帧先由神经 actor 产生 3 维世界坐标命令，然后 envelope 基于同一帧可测误差执行三个操作：

1. **死区稀疏化**：横向位置仍在合格中心死区内时，横向命令精确归零；
2. **方向门**：若命令会增大深度、横向或偏航误差，该分量归零；
3. **连续上限**：命令幅值不能超过对应连续恢复场的幅值，也不能超过原 actor 提议。

三个分量分别满足：

```text
abs(executed_i) <= abs(neural_proposal_i)
executed_i * desired_error_reduction_i >= 0
```

横向分量另外满足：

```text
abs(y) <= deadband  =>  executed_vy = 0
```

若 projection 结果比网络提议绝对值更大，运行时直接抛出错误，不会静默扩大权限。

这不是用手写控制器替换网络。网络仍在 56.72% 的全程帧上实际运行并决定有界幅值；envelope 是一种 shield，作用类似关节限位或 torque safety projection，只负责阻止已知无效方向和函数逼近残差。

## 3. 新增可审计回执

`shared_world` 从 v31/v15 升级为 goalkeeper config v32、result v16，并新增：

- envelope 是否启用；
- 每帧 raw neural world command；
- 每帧 envelope 后 world command；
- 每帧是否被调整；
- actor 激活帧中的调整比例；
- checkpoint hash、blend 和 actor 激活比例继续保留。

因此可以区分“模型没运行”“模型运行但被 shield 调整”“模型命令原样通过”三种情况。

## 4. 更严格的物理晋升门

仍运行四条球路，每条执行父策略一次、候选两次，共 12 次 25 秒三 G1 共享 MuJoCo 世界。在 S105 门控基础上新增：

- 每条球路候选峰值命令步长不高于父策略加 `1e-6 m/s`；
- 至少存在一条父策略零动作球路；
- 零动作球路候选 TV 不高于 `1e-6 m/s`；
- 每条球路必须观测到 envelope 调整；
- 运行结果必须声明 envelope 已启用；
- 组合 TV 仍必须低于父策略；
- 训练留出 `right-inner` 仍须单独通过。

没有放宽原来的高空扑救、腾空落地、双脚支撑、关节/力矩、pre-ready、successor probe、post-ready、前缀一致或严格复放门。

## 5. 最终结果

| 球路 | 父 TV | S106 TV | 父峰值步长 | S106 峰值步长 | 父 ready | S106 ready | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| left-inner | 0.413337 | 0.227008 | 0.091281 | 0.003931 | 7.706 s | 7.706 s | PASS |
| left-outer | 0.000000 | **0.000000** | 0.000000 | **0.000000** | 6.102 s | 6.102 s | PASS |
| right-inner（留出） | 0.126201 | 0.149176 | 0.091393 | 0.005070 | 4.706 s | **4.606 s** | PASS |
| right-outer | 0.306072 | 0.190027 | 0.092147 | 0.004728 | 6.002 s | 6.002 s | PASS |
| **组合** | **0.845609** | **0.566210** | — | 全球路不退化 | — | 最大退化 **0.000 s** | **PASS** |

组合 TV 比父策略下降 `33.04%`，也优于 S105 的 `29.53%`。S106 没有把 `right-inner` 的 TV 包装成逐球路胜利：它仍比父策略高 `0.022975 m/s`，但峰值跳变下降约 `94.45%`，并提前 `0.1 s` ready。新的全局门因此采用两条同时成立的要求：组合 TV 改善，并且每条球路峰值步长不退化。

envelope 调整比例很高（95.5%–100%）表示“至少一个分量被解析边界轻微裁剪的 actor 激活帧比例”，不表示 95%–100% 的神经输出被拒绝。实际 depth/yaw/lateral 命令仍可部分通过。

## 6. 证据与视频

### 正式物理证据

- 目录：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s106-context-gated-recovery-integration-v1`
- 报告：`evidence.json`
- report hash：`sha256:6ecc45f3e80009f587557e5e7ff5c6c0547ecd8e25facc7621427e1ee138a824`
- implementation hash：`sha256:9dc26bcf769d052312db235834bc96b28763b109061f5eed067f869df6e0cfe9`
- source commit：`e784438c8acfb3d4cc499f913f7e35b5dadb27b3`

### 1080p60 父子策略对照视频

- 视频：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s106-context-gated-recovery-showcase-v2/s106-context-gated-recovery.mp4`
- 时长：57.367 秒
- 分辨率：1920×1080，60 fps，3,442 帧
- video hash：`sha256:e18bc8ae11481e2b1aad5cae30eb2917097a170e21a0b0829a10965d1d022974`
- manifest hash：`sha256:b51db0ba4bddcbda490a517e70c28af792c3a5d4bb9aaafe0da44348f8b90ca6`

视频只负责展示，所有结论仍来自物理 telemetry，`pixels_used_for_scoring=false`。

## 7. 对 ROSClaw 的意义

这一轮提炼出一个比“G1 守门脚本”更通用的模式：

```text
learned proposal
  → causal context gate
  → monotone/non-expansive authority envelope
  → frozen qualified foundation controller
  → telemetry receipt
  → paired parent/candidate promotion exam
```

它适用于移动底盘、机械臂末端速度、无人机姿态修正等有界高层 residual：学习模块提供可塑性，解析 envelope 保持稳定性，父子策略考试决定是否晋升。这正是 Stability–Plasticity Dilemma 的一个工程化解法：允许学习，但学习不能越过已知安全和单调边界。

## 8. 下一步

S107 不应继续只优化同一条恢复轨迹。下一项高价值闭环是把“二次横移 probe”升级为真实二次来球：

1. 第一名前锋完成高空射门，门将真实扑出；
2. 球由第二名前锋、回弹板或受控传球形成二次射门；
3. 门将不能 reset/teleport，必须在第一次碰撞后自主恢复、重新观测并完成第二次手套扑救；
4. 训练数据同时记录第一次失败如何影响第二次动作；
5. 再引入低权限 residual actor-critic，而不是直接跳到关节力矩在线更新。

S106 仍然是 `SIM_ONLY`，不授权真机、ROS/DDS 或直接关节/力矩输出。
