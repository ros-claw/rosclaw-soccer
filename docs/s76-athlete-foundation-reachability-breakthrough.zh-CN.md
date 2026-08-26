# S76 Athlete Foundation V1：恢复可达性突破实施报告

日期：2026-08-24
状态：SIM_ONLY，未授权硬件，不能晋级部署
输入方案：`/code/rosclaw/rosclaw_football/rosclaw_soccer尝试突破.md`

## 结论先行

本轮停止了 S49--S75 式的残差门槛扫描，先问清楚“目标状态在当前动作空间里是否可达”。实验证明：继续放大旧残差不是突破口；真正的结构性问题之一，是恢复技能的**入口身体状态与动作参考相位不匹配**。

在同一份 256 状态失败库、同一组四卡、同一组 seed（7620--7623）上，只改变“Get-up 是否按当前本体状态匹配参考相位”，得到严格配对结果：

| 路由 | 固定从第 0 帧进入 | 本体状态相位匹配 | 改善 |
|---|---:|---:|---:|
| Athlete / 动量吸收与站稳 | 217/225（96.44%） | 217/225（96.44%） | 0 |
| Get-up / 独立恢复专家 | 0/31（0%） | 31/31（100%） | +31 |
| 总体 | 217/256（84.77%） | 248/256（96.88%） | +31 |

逐状态比较中改善 31 例、退化 0 例。程序给出的硬判定为：

`PHASE_ENTRY_ADAPTER_BREAKTHROUGH_CONFIRMED`

这不是“又调好了一组阈值”。它把 rosclaw 的能力从“调用一个技能”推进为“根据本体状态选择专家，并选择专家内部合适的进入相位”。该机制可用于起身、扑救、踢球、抓取、工具操作等所有长时技能。

## 1. 为什么旧路线长期没有突破

### 1.1 256 状态 Recovery Reachability Bank

从三份不同 seed 的精确 MJX 失败记忆中，按来源、事件窗口、姿态代理和角速度代理分层抽取 256 个不重复状态。每个状态绑定 qpos、qvel、上一动作、教师动作、残差、历史本体感和轨迹相位。

失败库真实分布为：

- `UPRIGHT`：225；
- `KNEELING_OR_SUPPORTED_PROXY`：31；
- 角速度 LOW / MEDIUM / HIGH：100 / 150 / 6；
- 来源分布：81 / 87 / 88；
- 源库没有记录接触拓扑，因此 `stratification_complete=false`。

这个发现很重要：此前所谓“恢复失败”主要是**仍站立但有危险动量的 Capture/Absorb 失败**，并不是一份覆盖仰卧、俯卧、侧卧、跪姿和各种接触拓扑的完整起身数据集。把这些问题全部塞给同一个 bounded residual actor，本身就是任务定义错误。

### 1.2 1× / 4× 残差权威反事实

在完全相同的 256 状态上运行旧 bounded residual 和只用于诊断的 4× residual：

- 1× candidate：0/256；
- 4× candidate：0/256；
- 1× 平均关节目标增量仅约 `1.92e-5 rad`；
- 4× 平均增量约 `7.67e-5 rad`，确实接近四倍，但没有产生成功；
- parent 在不同进程的严格门槛附近出现 0--1/256 的 GPU 数值抖动，因此今后的权威反事实必须 lockstep。

硬路由结论是 `TRAIN_PARENT_FREE_EXPERT_ORACLE`。这正式终止了“再扫一轮 residual budget/gate”的主线。

## 2. 新架构：Athlete Foundation + Reachability Router + 专家

本轮实现的最小闭环是：

```text
本体状态
  -> Reachability Router
     -> UPRIGHT + 高骨盆：Athlete / 循环步行站稳策略
     -> 其他：Get-up 独立神经专家
        -> Skill Entry Adapter（参考相位估计）
  -> 双足 + 低线速度 + 低角速度连续保持
  -> 预热循环步行策略
  -> 平滑交权
  -> LOCOMOTION_READY proxy
```

成功不再是“骨盆高度越线一瞬间”。每个状态必须在 20 秒物理闭环中完成：双足支撑、低动量连续保持、下游循环运动策略预热、平滑接权，以及末尾至少连续 2 秒稳定。

### 2.1 数据驱动 Skill Entry Adapter

旧运行时把任何失败状态都强行送入 Get-up 参考动作第 0 帧。参考动作第 0 帧骨盆高度约 0.06 m，而 31 个路由状态的骨盆高度约 0.64--0.66 m；机器人本来接近站起，却被命令重新执行卧地动作，因而 0/31 失败。

新的入口适配器只使用进入时可获得的本体状态，对 450 帧参考运动计算：

```text
J(f) = MSE(q, q_f)
     + 0.05 * MSE(qdot, qdot_f)
     + 2.0 * (h - h_f)^2
     + 0.5 * (1 - |<r, r_f>|^2)
```

其中 `q/qdot` 是 29 关节位置和速度，`h` 是骨盆高度，`r` 是骨盆四元数；四元数距离对正负号不敏感。所有权重和每个状态选择的参考帧都写入哈希绑定报告。

31 个 Get-up 状态自动匹配到第 435--438 帧，主要是第 436 帧。这不是人工为某个案例写补丁，而是从运动数据中按当前身体状态选择技能相位。

## 3. 三层证据

### 3.1 专家本域对照

同一个 Get-up 权重在自己的参考起点域、加入 0.12 初始姿态/速度扰动时：

- 28/28 开始交接；
- 28/28 完成交接；
- 28/28 满足末端连续稳定；
- 每例末尾连续稳定 7.50--9.12 秒。

因此权重本身没有损坏。固定第 0 帧在真实失败状态上 0/31，属于前驱状态/动作相位鸿沟。

### 3.2 四卡无相位匹配基线

四张 A6000 各运行连续 64 状态：

- Athlete：217/225；
- Get-up：0/31；
- 总体：217/256；
- Get-up 交接完成：0/31。

这也验证了为什么不能只看总体 84.77%：优势 strata 的微平均会掩盖某个专家 0% 的事实。聚合器现以最弱的非空路由控制架构结论。

### 3.3 seed 绑定的相位匹配配对 A/B

四张卡、状态切片和随机种子与基线完全一致，只打开 `getup_reference_phase_alignment`：

- Athlete 逐状态结果完全不变：217/225；
- Get-up：31/31；
- 总体：248/256；
- 所有 256 状态都完成专家或 Athlete 交接，8 个 Athlete 状态在末端稳定门失败；
- A/B：31 改善，0 退化。

证据哈希：

- 256 状态失败库：`sha256:ea631761ae3ecf2a920832081eced32e1b432c0146d4e9810bfb6dbe43c61325`；
- seed 绑定基线聚合：`sha256:120e0b3e783034ee96e72272e2ee1841ec19a6142cfb406220ddb2c49b4c7eff`；
- seed 绑定相位匹配聚合：`sha256:d132a4b37922fb42035c2b7131ce4ad8f1a355d29dc315eba34f0d7a501fcaa6`；
- 配对 A/B 决策：`sha256:08dc4cef7ea919addb4e6facc749c1f8715567f7b558d63b42308448b5494a4c`。

## 4. 工程实现

### 4.1 通用 rosclaw / growth 能力

- 256 状态精确失败记忆合成与去重；
- failure-state 1× / 4× residual authority 诊断；
- `ReachabilityMainline` 硬决策，不允许坏结果继续扫门槛；
- Hybrid MoE 状态路由；
- 数据驱动技能入口相位估计；
- 双足低动量连续保持与循环策略预热交权；
- 四卡 state/seed/device/content hash 聚合；
- 按路由报告，禁止总体平均掩盖专家失败；
- 精确逐状态 A/B，拒绝状态、seed、设备或物理契约不一致的比较。

### 4.2 MuJoCo/MJWarp 加固

没有使用 Isaac Gym。现代 MuJoCo 3.10 将 `MjSpec.detach_body` 改为通用 `delete`，并改变了 joint damping 的 Python 绑定；兼容层已同时在 MuJoCo 3.3.1 和 3.10.0 编译同一球场模型，均得到 32 bodies / 138 geoms。

失败库只包含 G1 的 36 qpos / 35 qvel，而足球场景追加球的 7 qpos / 6 qvel。跨场景重放现在只精确覆盖 canonical G1 前缀，保留目标场景足球状态，并明确记录跨场景边界。

### 4.3 测试

- 目标回归：32 passed；
- S76 新模块：8 passed；
- Ruff format/check：通过；
- MuJoCo 3.3.1 / 3.10.0 双版本球场编译：通过；
- 四张 A6000 均绑定为 `cuda:0..3`，没有 CPU 回退。

## 5. 必须诚实保留的边界

这次是恢复**可达性和技能入口**突破，不是完整足球门将晋级：

1. 256 状态旧失败库中 225 个仍是直立状态，31 个是跪/支撑代理；没有完整覆盖 prone、supine、left/right side。
2. 源失败库没有接触拓扑，不能证明手、膝、躯干与地面的真实接触分层。
3. 状态从 OpenTrack G1 恢复模型重放到足球场 MJWarp 模型，虽然 qpos/qvel 和身体契约绑定，但仍是 cross-scene diagnostic。
4. 当前 successor 是 `LOCOMOTION_READY proxy`，还不是 `GOALKEEPER_READY`，没有证明第二脚来球可继续扑救。
5. 8 个失败都在 Athlete 路由，说明动量吸收/Capture 专家仍需训练。
6. 本证据 `promotion_eligible=false`、`hardware_command_sent=false`，不能用于真实机器人。
7. 因为还没有通过完整 Flying Save -> Recovery -> Second Save 的物理真值门，本轮没有用宣传视频替代科学证据。

## 6. 下一轮硬任务

### P0：补全真实后继分布

在 CPU MuJoCo 物理真值中重新采集扑救后失败库，记录接触 pair、接触冲量、支撑多边形、COM、脚/膝/手/躯干接触、前驱扑救动作和 recurrent history。按 prone、supine、左右侧卧、跪姿、直立高动量分层。

### P0：攻克剩余 8 个 Athlete 状态

训练 parent-free privileged Capture/Absorb PPO oracle，而不是继续扩大 residual：

- 输入完整本体感、接触、COM/支撑关系和短历史；
- 输出 29DoF PD target，后续再比较 torque policy；
- 以动量耗散、无后退、双足低冲击接管为目标；
- oracle 先在失败状态达到 70%，再做 history-aware distillation。

### P1：让入口适配器可学习、可拒绝

当前最近参考相位是强基线。下一步加入 phase confidence、时序历史和 OOD 拒绝：低置信度时进入 Capture 专家或 parent-free recovery oracle，不能把陌生姿态硬塞给 Get-up。

### P1：完整门将闭环

按方案执行 `Flying Save -> contact-rich post-dive state -> Capture/Get-up -> GOALKEEPER_READY -> second shot`。必须在固定盲测集上分别报告远角低/中/高球，不能用回中作为唯一成功条件。

### P2：视频

只有完整闭环通过后，再录制带 state id、seed、policy hash、逐球物理结果和失败案例的长视频。视频是可视化证据，不是对失败 gate 的替代。

## 7. 证据位置

- Reachability bank：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/reachability-bank-256-v1/`
- 1× / 4× audit：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/reachability-audit-v1/`
- 无相位 MoE：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/recovery-moe-probe-v1/`
- 专家本域 28 例：`recovery-moe-probe-v1/source-domain-control-28-gpu0.json`
- seed 绑定配对 A/B：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/recovery-moe-seeded-ab-v1/`
- 最终配对决策：`recovery-moe-seeded-ab-v1/paired-ab-decision.json`

本轮最核心的工程教训是：**技能权重、身体状态和技能相位必须共同构成可达性契约。** 一个会起身的模型，如果从错误的动作相位进入，也会表现得像完全不会起身。rosclaw 的 growth 不应只会更新权重，还必须学会“何时进入哪一个技能、从技能的哪里进入、何时安全交给下一个技能”。
