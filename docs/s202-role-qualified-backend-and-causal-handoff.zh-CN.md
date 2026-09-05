# S202：角色专用后端再认证与连续链学习交棒

日期：2026-09-06

状态：`PASS_ROLE_BACKEND_AND_CAUSAL_HANDOFF` / `SIM_ONLY`

物理权威：当前代码 CPU MuJoCo 双次严格回放；视频像素不参与评分

## 结论先行

S201 证明了六个独立 ROSClaw cell 可以把坏传球只归因给 playmaker，但其单样本
瞄准 residual 不能部署。S202 没有继续扩大这个脆弱补丁，而是复查历史训练资产，
找到已经通过多上下文训练的角色专用 `dynamic_lead_pass` actor，并把它接入新的
role-qualified backend router。

新鲜物理再认证结果：

| 指标 | S202 结果 | 门槛/解释 |
|---|---:|---|
| 传球交付误差 | `0.00478 m`（`0.48 cm`） | `≤0.05 m` |
| 传球峰值球速 | `1.518 m/s` | `≥1.0 m/s` |
| 接球停顿帧 | `0` | 一脚接射，不等球 |
| 射门峰值球速 | `8.228 m/s` | `≥4.0 m/s` |
| 射门目标误差 | `1.115 m` | 未通过 `≤0.10 m` |
| 传球者最低骨盆高度 | `0.681 m` | 安全 |
| 射手最低骨盆高度 | `0.659 m` | 安全 |
| 门将最低骨盆高度 | `0.750 m` | 安全 |
| 机器人互撞 | `0` | 通过 |
| 严格重复回放 | 完全一致 | 结果对象与轨迹 digest 均一致 |

因此本轮不是“全链成功”。传球后端已经通过，连续链真实完成 `PASS → RECEIVE`，
随后在 `SHOOT` 精度上失败。信用分配器把下一轮唯一可塑角色从 `red.playmaker`
交棒给 `red.finisher`；门将没有接触到这记偏离球门的射门，不能把失败算给门将。

## 为什么不是继续用通用任意球策略

S200 的独立 playmaker 虽然触球且安全，但交付误差为 `1.284 m`。本轮新增的
`RoleOptionBackendCandidate` 不看名字或演示次数，而要求候选同时满足：

1. agent 的本体角色拥有对应技能；
2. backend 明确服务该 option；
3. 足够多的不同上下文和不同物理轨迹；
4. 严格回放、封存 holdout、父策略保留全部通过；
5. `SIM_ONLY`，无直接关节力矩或硬件权限。

即使给 `generic_freekick` 填入 100 个上下文、100 条轨迹，它也不能成为 `PASS`
后端；它缺少角色/option 语义授权。`dynamic_lead_pass` 则有 8 个不同训练上下文、
8 条不同物理轨迹、两个通过的封存留出场景，而且留出候选都胜过冻结父策略，故
获准成为 playmaker 的传球后端。

这是一项通用 ROSClaw 能力：不同任务的智能体不能仅凭“动作看起来相近”复用任意
policy，必须同时满足角色能力、证据域和动作接口约束。

## 一次重要的 Stability–Plasticity 失败

第一次再认证曾把动态传球 actor 外推到团队默认的 `2.19 s` 接球启动相位。交付
误差进一步降到 `0.00445 m`，但改变后的到球位置使射手触发关节边界违规。门控据此
拒绝整份报告，没有因为传球更准而忽略下游身体风险。

最终版本回到训练时未泄漏的封存困难格：启动相位 `1.92 s`、横向跑位 `+0.10 m`。
当前代码重新执行两次，严格轨迹 digest 都是
`sha256:cea35a48f27f6efb6eb89514531ed167381f78893393bcf2d49eba0f9ab0f86a`，
并保持全队安全。这里的原则是：成熟能力可以再认证，不能跨出已验证支持域后仍沿用
原证书。

## 连续同球因果链

S202 从物理轨迹提取同一足球的连续状态哈希，形成三个有序事件：

```text
red.playmaker PASS
  └─ red.finisher RECEIVE（one-touch readiness，无伪造额外触球）
       └─ red.finisher SHOOT
            └─ 目标误差 1.115 m → SHOT_INACCURATE
```

每个事件绑定输入/输出球状态、角色 cell、Champion/Parent、决策、请求、时间和严格
回放状态。最终 assessment：

- `completed_phase_count=2`；
- `ball_lineage_verified=true`；
- `safe=true`；
- `exact_replay=true`；
- `earliest_failure=shot_inaccurate`；
- `focal_agent_id=red.finisher`；
- `downstream_credit_blocked=true`。

新的 `PlasticityLease` 仍覆盖六个独立 cell，但只有 `red.finisher=PLASTIC`，另外五个
全部 `FROZEN`。这就是“球队一起成长”不互相污染的最小闭环：成功模块被冻结保留，
训练预算自动移交给最早失败模块。

## 代码产物

- `growth/continuous_option_chain.py`：同球状态机、最早失败归因及角色局部租约。
- `growth/pass_aim_calibration.py`：上下文/轨迹去重的鲁棒瞄准 residual actor。
- `growth/role_option_backend.py`：角色和证据双重限定的 option backend router。
- `growth/runtime_finish_plan_actor.py`：把 passer yaw 按周期角处理，消除 `-π/+π`
  表示跳变造成的虚假 OOD；真实超域仍保持拒绝。
- `training/independent_chain_credit_growth.py`：S200 失败到 S201 训练 seed 的闭环。
- `training/role_backend_continuity_evidence.py`：S95 资产适配、当前物理再认证和 S202
  学习交棒。
- `media/independent_chain_credit_video.py`、`media/role_backend_continuity_video.py`：
  证据下游诊断视频；不能用于评分或晋升。

## 外部冻结证据

```text
$ROSCLAW_SOCCER_EVIDENCE/athlete-foundation-v1/
  s201-independent-chain-credit-v4/
  s202-role-backend-continuity-v5/
```

S202 关键文件：

- `role-backend-continuity.json`：report hash
  `sha256:7a8a9e5fd00a247a255c50afc4493e1859a3bb06c7019b24cd9d94807317aa25`；
- `routed-continuity-trajectory.npz`：新鲜物理轨迹及 replay digest；
- `s202-role-backend-causal-handoff-final-1080p.mp4`：36.97 秒、1920×1080、30 fps；
- 同名 `.json`：视频、报告和轨迹的哈希绑定。

视频只是可读性层。`promotion_eligible=false`、`pixels_used_for_scoring=false`，物理
结果与学习交棒均来自数值轨迹和接触事件。

## 诚实边界与下一轮

本轮新鲜场景有三台物理 G1；六 cell 是独立成长与冻结租约的完整球队范围，并不代表
六台机器人都在这条新轨迹中同时运动。传球已到 0.48 cm，但射门未进、门将未扑；
所以还不能声称 `PASS → RECEIVE → SHOOT → SAVE` 全链成功。

S203 应只训练 finisher 的角色专用 `RUNTIME_FINISH_PLAN`：输入必须包含动态来球状态
和接球相位，先在与 discovery 隔离的多个 goal-corner holdout 上达到 `≤0.10 m`，
同时保持零停顿、无关节违规、父策略传球能力不退化。只有射门越过精度门后，才把同一
颗球继续交给 goalkeeper 的 `VISIBLE_BALL_GOALKEEPER`，再训练真正的扑救而不是让
门将为偏出球门的射门背锅。

### S203 前置诊断：先消除假 OOD，再承认真 OOD

把 S189 离散父 actor 与 S195 continuous critic 接到 S202 来球时，两者最初都报告
支持距离约 `617.47`。逐特征检查发现，训练记忆把接近 `π` 的身体朝向存成负角，
新请求把同一侧朝向表示成正角；线性距离错误地把这两个几乎相邻的姿态看成相差
`2π`。运行时现已用圆周最短角距离消除这一坐标伪影，并增加正负 `2π` 等价测试。

修复后支持距离降至 `12.315`，但仍高于准入上限 `2.75`，所以 actor 继续正确返回
`RUNTIME_FINISH_PLAN_OOD_FALLBACK`。剩余主要真实距离贡献为：receiver lane
`9.375σ`、passer yaw `6.979σ`、球位 y `2.839σ`、球位 x `2.293σ`。这表明下一轮
curriculum 必须联合扩展“接球横向跑位—传球者朝向—球位”，不能仅调一个射门目标，
更不能把 `maximum_support_distance` 调大来绕过证据。
