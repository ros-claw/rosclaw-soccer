# S199：每台 G1 背后的独立 ROSClaw 与 3v3 自治基础

日期：2026-09-05

证据上限：`SIM_ONLY`

## 结论

S198 的确没有达到“每个球员都有自己的心智”。它仍是三台 G1 共用一个世界级状态机，
只给两个非持球角色叠加了有限目标修正。S199 已完成第一步架构替换：同一 MuJoCo
球场中有 6 台完整 29-DoF G1，红蓝各 3 台；每台身体都绑定一个不同的 ROSClaw
Agent Cell，分别拥有自己的角色自我模型、策略身份、个人记忆、失败记忆、能力档案和
`IndividualGrowthScope`。队内协作不再靠共享脚本暗中安排，而是先各自观察和决策，
再通过有哈希承诺的 `TeamCoordinationFrame` 协商。

四个 fresh retention 病例及其独立重放全部通过。24 个“机器人 × 病例”轨迹中，平均
主动运动比例为 **97.85%**，最低骨盆高度为 **0.7447 m**，最大身体倾斜为
**0.2042 rad**，没有关节限位违规、力矩越界或机器人互撞。

不过这还不是一场完整自主比赛。当前已经打通的是“独立观察—角色决策—队内协商—稳定
全身移动”；`PASS / SHOOT / SAVE / DISTRIBUTE` 仍是高层意图，尚未全部路由到项目中
已有的物理传球、射门和扑救 option。证据和视频都明确记录
`contact_skill_router_complete=false`，不能把本阶段宣传成完整传射扑闭环。

## 1. “独立 ROSClaw”具体是什么意思

每台 G1 现在都有以下独立状态：

- `RoleSelfModel`：知道自己是哪支队、是门将/组织者/终结者、队友和对手是谁、允许使用
  哪些技能；
- `AgentCellObservation`：只消费自己的本体状态、球、球门、队友和对手的数值观测，不用
  像素评分或特权标签；
- 独立策略 artifact hash：六名球员可以拥有不同代际和不同策略；
- 独立 `personal_memory_namespace` 与 `failure_memory_namespace`：失败不会混入另一个
  球员的职业记忆；
- 独立 `IndividualGrowthScope`：Parent、Candidate、Champion 和生涯 lineage 都按
  `agent_id` 绑定；
- 独立 `PlasticityLease`：每次只允许一名球员更新，另外五名球员必须冻结。测试证明一旦
  冻结球员的策略 hash 被改动，审计会拒绝该轮学习。

六个 Agent Cell 当前在同一个仿真进程和同一个物理时钟中运行，而不是启动六个操作系统
级 `rosclawd`。这是有意的：它们在身份、记忆、策略和可塑性上相互隔离，但必须共享同一
足球、碰撞求解器和同时钟，才能检验真实协作。将“独立”错误地实现成六个互不相见的仿真
进程，反而无法形成一场物理比赛。

```text
                 同一个 CPU MuJoCo 世界 / 同一颗足球
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
  red.goalkeeper         red.playmaker          red.finisher
  private GrowthScope    private GrowthScope    private GrowthScope
  private memories       private memories       private memories
        │                      │                      │
        └────────── 各自意图 ─┴─ 显式握手 ──────────┘
                               │
                     TeamCoordinationFrame
                               │
              50 Hz 冻结神经步态 / 500 Hz 关节反馈

  蓝方以完全相同的结构拥有另外三个独立 Agent Cell。
```

## 2. 队伍与角色

正式验证是 3v3：

| 球队 | 球员 | 职责 | 可选择意图 |
|---|---|---|---|
| 红 | `red.goalkeeper` | 门将 | COVER / SAVE / DISTRIBUTE |
| 红 | `red.playmaker` | 组织者 | RECEIVE / SUPPORT / CARRY / PASS |
| 红 | `red.finisher` | 终结者 | RECEIVE / RUN_IN_BEHIND / SHOOT |
| 蓝 | `blue.goalkeeper` | 门将 | COVER / SAVE / DISTRIBUTE |
| 蓝 | `blue.playmaker` | 组织者 | RECEIVE / SUPPORT / CARRY / PASS |
| 蓝 | `blue.finisher` | 终结者 | RECEIVE / RUN_IN_BEHIND / SHOOT |

多 G1 场景构建器并不锁死在 3v3：它支持 2–10 台独立 29-DoF G1，即当前基础设施可继续
扩展到 5v5。正式保留集仍只宣称已验证 3v3；在 5v5 经过安全、重放和角色信用验证前，
不会提前宣称扩队成功。

## 3. 实际物理闭环

每 0.1 秒，六个 Cell 分别从同一物理快照构造自己的 egocentric observation，并独立
产生一条角色许可内的决策。传球者不能单方面宣布传球：接球者必须同时选择
`RECEIVE / SUPPORT / RUN_IN_BEHIND`，系统才会签发一个包含双方观测 hash、策略 hash、
落点和到达窗口的 `PassReceiveHandshake`。

高层决策只输出战术意图和二维目标，无权写 root pose、关节或力矩。目标经过速度、
加速度、到达半径和球员间距约束后，由每台 G1 自己的 RoboNaldo 神经步态执行：

- 神经策略：50 Hz；
- 高增益关节 PD：500 Hz，每个物理子步重新闭环；
- 受预测关节限位保护和 85% 硬力矩包络约束；
- 初始化后不写机器人 root pose；
- 初始化后不写足球位姿或速度；
- 视频只是通过轨迹的下游重放，不参与评分。

## 4. 本轮发现并修复的真实问题

### 4.1 六台 G1 同时摔倒

首版把 50 Hz 神经策略目标算出的 PD 扭矩保持了整整 20 ms。对 G1 的高增益关节来说，
这不是一个稳定的闭环，启动峰值扭矩约为既有稳定路径的两倍，六台机器人在 1–2 秒内
全部倒地。

修复后，actor 仍以 50 Hz 更新关节目标，但 PD 在 500 Hz 的每个 MuJoCo 子步基于最新
关节位置和速度重算扭矩。相同场景随即从“六台全倒”变为六台稳定、零碰撞。这是运动控制
时序错误的修复，不是降低跌倒判据或锁住身体。

### 4.2 蓝方把守门员当成前插接球人

接球人筛选最初隐含了“进攻方向总是 +x”的红方假设。复制给蓝方后，蓝方组织者会把
身后的门将选为接球人；门将正确拒绝接球，整个 coordination frame 因此 fail-closed。

现在进攻方向由“自身位置 → 对方球门”实时推导，红蓝完全对称；专门的回归测试保证蓝方
组织者会选择前方终结者，而不是门将。

### 4.3 非法初始重叠没有被美化成成功

第一版射门保留病例把球生成在红方终结者双脚之间。尚未接入射门 option 的 locomotion
发生非预期触球，机器人触发关节安全失败，整个 v2 报告状态为
`REJECTED_3V3_FOUNDATION`。该失败证据保留在外部目录。v3 将球放入有球权但无几何穿插
的准备区域，重新从零运行并通过；没有放宽安全门，也没有删除失败来改写结果。

### 4.4 初始速度/加速度过激

早期 0.58 m/s、1.40 m/s² 参数导致策略启动扭矩和身体摇摆偏大。最终运行包络采用
0.38 m/s 场上速度、0.30 m/s 门将速度和 0.60 m/s² 加速度，并加入 0.95 m 球员间隔
盾。它不是最终比赛速度，而是独立多智能体基础首次晋级时的稳定 Champion；之后只能由
带父代回归的 Growth 逐步放宽。

## 5. 正式 fresh retention 结果

每个病例从时间零完整运行两次，结果对象和完整轨迹摘要必须一致：

| 病例 | 传球握手 | 射门意图 | 扑救意图 | 门将分发 | 世界通过 | 精确重放 |
|---|---:|---:|---:|---:|---:|---:|
| 红方推进 | 12 | 0 | 5 | 0 | 是 | 是 |
| 蓝方反击、红门将防守 | 2 | 0 | 2 | 48 | 是 | 是 |
| 终结者射门选择 | 0 | 50 | 0 | 0 | 是 | 是 |
| 蓝门将分发、双方重组 | 18 | 0 | 15 | 32 | 是 | 是 |
| **合计** | **32** | **50** | **22** | **80** | **4/4** | **4/4** |

全局质量：

| 指标 | 结果 |
|---|---:|
| 独立 Agent Cell | 6 |
| 独立个人记忆 / 失败记忆 | 6 / 6 |
| 独立单焦点 Plasticity Lease | 6 |
| 机器人 × 病例 | 24 |
| 平均主动运动比例 | 97.85% |
| 平均位移 | 1.3142 m |
| 最低骨盆高度 | 0.7447 m |
| 最大身体倾斜 | 0.2042 rad（约 11.70°） |
| 机器人互撞 | 0 |
| 关节限位 / 力矩越界 | 0 / 0 |

正式状态：`PASS_INDEPENDENT_3V3_FOUNDATION`

## 6. 代码、证据与视频

主要实现：

- `growth/independent_agent_cell.py`：独立 Cell、自我模型、决策、协商、单焦点可塑性租约；
- `skills/team/independent_team_world.py`：同钟六 G1 世界、双速率控制、轨迹和质量证据；
- `training/independent_team_growth.py`：3v3 roster、四类保留集、独立重放和完整性验证；
- `world/multi_player.py`：2–10 台独立 G1 的共享球场构建器，并复用冻结的
  `world/field.py` 球场基座；
- `media/independent_team_video.py`：只消费通过轨迹的六 G1 可视化。

正式证据目录：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s199-independent-3v3-v8/`

- 报告：`retention-exam.json`
- 报告内生 hash：`sha256:2fe4d155afdc58a70fb76f0da8a63fb374dea94161bfeeb3eef57cbf43cc463a`
- 报告文件 hash：`sha256:5066b9eced385efa66e48bc0a329d8e90eb56771a0620fe346d31c76117dc59e`
- 22.67 秒、1920×1080、30 fps 视频：
  `video/rosclaw-independent-3v3-s199-v8-1080p.mp4`
- 视频 hash：`sha256:e3d4be7a61a04dfa9aa5aa2997c50180b4d2cd2fff50c5bc2ff8df005cbd2528`

外部 RoboNaldo 资产的商业宣传许可仍未证明，因此视频清单保持
`commercial_use_allowed=false`。它可以作为本地研发审阅材料，不能据此宣称已取得商业
传播授权。

## 7. 下一阶段：从“各自会想”走向“真的会比赛”

S200 的首要工作不是继续增加走动人数，而是完成可验证的物理 option router：

1. 把 `PASS` 路由到已有的传球接触技能，并要求传球握手、实际足部接触、球速增益和接球
   事件处于同一个不重置物理时钟；
2. 把 `SHOOT` 路由到目标条件化射门 option，并让终结者自主选择射门或回传；
3. 把 `SAVE` 路由到现有门将移动/起跳/双手扑球技能，把 `DISTRIBUTE` 路由到安全出球；
4. option 结束后必须回到持续角色决策，不得以“事件完成”终止整场；
5. 先形成 3v3 连续事件链，再扩为 4v4/5v5。增加人数前必须证明碰撞、计算预算和角色信用
   仍可审计；
6. 状态机仅作为 warm-start teacher，采集成功与失败后训练 centralized-training /
   decentralized-execution 的角色 actor；共享足球表征，但每名球员保留个人 adapter、
   失败记忆和 Champion；
7. 每轮只发放一个 `PlasticityLease`，按传球质量、无球跑动、封堵、扑救、恢复和阵型贡献
   做角色级信用分配。候选必须同时通过个人技能、队内兼容、父代回归和困难域 fresh blind。

因此本轮的实质突破是：球队第一次不再是“一只手操纵多个木偶”，而是六个可独立学习、
可独立失败、必须显式协商的 ROSClaw 个体。下一轮要解决的是让这些个体调用已经训练好的
身体技能，形成不中断的真实传—射—扑比赛链路。
