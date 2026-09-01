# S118：从技能片段走向连续比赛、First Touch 与团队成长

日期：2026-09-01

证据上限：`SIM_ONLY`

实现分支：`codex/s118-continuous-soccer`

## 结论先行

`rosclaw_soccer想法1.md` 的方向成立，但不能直接从现有三角色 Demo 跳到大规模
5v5。当前最短、最能证伪的路线是：

1. 先建立不能被片段拼接伪造的 `ContinuousSoccer` 时间线；
2. 再把移动来球的第一脚处理做成独立、可测量、可做梦的 Growth 任务；
3. 冻结 Athlete Foundation 和现有传/射/恢复技能，只训练 2v1 的高层传射决策；
4. 这些层通过后，才扩大到 2v2、历史对手池和球队级 Match Dream。

S118 基础阶段完成了前三项的**合同与证据基础**，并跑通官方 `dm_control` 的 60 秒
连续 MuJoCo 环境语义。后续 S118-A 已进一步完成单场景 G1 First Touch 物理获取、
确定性重放和配对考试，但尚未训练出 First Touch Champion，也尚未宣称 G1 已学会
2v1。局部结果和泛化失败详见
`docs/s118a-first-touch-acquisition.zh-CN.md`。

## 为什么现在不能直接上大球队

现有项目已经拥有大量低层运动、射扑和恢复代码，但多数证据仍围绕预设技能链。
如果此时让多个 G1 从 29 关节动作一起做 MARL，会同时改变：

- 身体稳定性；
- 传球和射门技能；
- 高层战术；
- 对手分布；
- 奖励归因。

最终即使球队得分提高，也无法知道是战术成长、某个球员变强，还是对手退化。
因此 S118 固定三层频率和权责：

```text
Tactical policy       5–10 Hz   只决定 PASS / SHOOT / DRIBBLE / HOLD
Skill policy         25–50 Hz   执行接、传、射、跑、恢复
Athlete Foundation   25–50 Hz   平衡、步态、全身控制与个人适配
```

高层候选没有权限修改低层策略哈希；所有反事实比较必须复用相同初态、种子、环境、
身体基础和技能 bundle。

## 外部研究与本地参考审查

### DeepMind：层级策略与连续比赛

`From Motor Control to Team Play in Simulated Humanoid Football` 的关键不是单一
射门动作，而是人类动作先验、中层足球技能、MARL 和 population learning 的分阶段
组合。官方 `dm_control` soccer 环境还明确支持 `terminate_on_goal=False`：进球后
场内重开，同一个 Episode 继续到时间上限。

本轮下载并冻结：

- 仓库：`/code/rosclaw/rosclaw_football/repos/dm_control`
- commit：`d2cf3a3d3ad47a9ea4153710fca852375adb4dff`
- 许可：Apache-2.0

没有复制其实现代码；ROSClaw Soccer 只把它作为外部、可选的物理环境合同探针。

### HumanoidSoccer / PAiD：G1 移动球动作先验

本地官方仓库：

- 路径：`/code/rosclaw/rosclaw_football/repos/HumanoidSoccer`
- commit：`e72e470230047dedaf66df0983f1d0ab746faeb5`
- 许可：CC BY-NC 4.0，只能作为非商业研究参考，不能进入商业宣传资产。

它提供 G1 动作数据、移动球感知动作融合和 MuJoCo sim2sim。它适合做 First Touch
教师/运动先验来源，但当前 rollout 只记录触球、方向和最大球速，缺少 First Touch
晋级所需的接球目标区、下一动作延迟、骨盆高度和躯干倾角，所以不能把它的“移动球
射门”直接声明成“控好第一脚”。

### ICRA 2026 Agile Striker：训练结构，不采用旧仿真器

本轮下载并冻结：

- 仓库：`/code/rosclaw/rosclaw_football/repos/agile-striker-icra2026`
- commit：`378a12ac7446cd175f973c04e32912eb9acbee10`
- 许可：Apache-2.0（另含上游组件许可要求）

可借鉴的是四阶段训练结构：特权教师、定向踢球、DAgger 学生、感知噪声下的约束
RL，以及 10 Hz 感知 / 50 Hz 控制和主任务/代价多 critic。其运行环境是旧 Isaac
Gym，本项目不复用该运行栈；这些算法思想将移植到 MuJoCo/MJX 训练后端。

### 主要研究来源

- DeepMind, [From Motor Control to Team Play in Simulated Humanoid Football](https://arxiv.org/abs/2105.12196)
- Google DeepMind, [dm_control soccer 环境源码](https://github.com/google-deepmind/dm_control/tree/main/dm_control/locomotion/soccer)
- Google DeepMind, [Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning](https://deepmind.google/research/publications/31284/)
- Daffan et al., [Agile Humanoid Striker 论文](https://arxiv.org/abs/2512.06571) 与
  [官方代码](https://github.com/Daffan/humanoid-soccer)
- TeleHuman, [HumanoidSoccer / PAiD 论文](https://arxiv.org/abs/2602.05310) 与
  [官方代码](https://github.com/TeleHuman/HumanoidSoccer)

## 本轮代码实现

### 1. `ContinuousSoccer` 证据合同

新增 `skills/team/continuous_soccer.py`：

- 固定 30–180 秒连续时钟；
- 高层频率必须低于技能层频率；
- `terminate_on_goal` 和 `reset_clock_on_goal` 必须为假；
- 所有事件必须来自物理状态，视频像素不得参与评分；
- 事件 ID 唯一、时间单调；
- 至少有两名球员且来自两支不同球队；
- 进球后下一个事件必须是同 Episode 的 `RESTART`；
- `RESTART` 后还必须发生新的足球事件；
- 首尾及相邻活动事件间隔不得超过 10 秒，防止“踢一脚后发呆”也冒充连续比赛；
- 只接受 CPU MuJoCo strict replay，且永远 `SIM_ONLY`。

### 2. First Touch 因果失败与 Dream

新增 `growth/first_touch.py`，测量：

- 来球/出球速度；
- 第一脚目标区误差；
- 出球方向误差；
- 左右脚选择；
- 骨盆最低高度、躯干最大倾角、根部最大速度；
- 从第一脚到下一次传/带/射动作的延迟。

稳定失败优先级为：

```text
未触球/触球太软
错误选脚
失去平衡
错误方向
触球太重
下一动作太慢
```

同一个样本会保留所有失败，但只用稳定主因路由 Growth。失败会调用 ROSClaw Core
通用 `FailureConditionedDream`，在训练域内生成：来球速度 ±10%、角度 ±10°、摩擦、
选脚和身体速度扰动。Dream 不具备晋级或硬件权限。

### 3. 2v1 反事实贡献合同

新增 `growth/tactical_2v1.py`：

- 高层动作只有 `PASS / SHOOT / DRIBBLE_LEFT / DRIBBLE_RIGHT / HOLD`；
- 状态绑定 frozen Athlete Foundation、skill bundle 和 Defender 快照；
- 奖励实现 `team + role + counterfactual + progress` 四项；
- counterfactual 必须是同初态、同环境、同种子的“移除焦点球员”独立物理回放；
- 原回放和消融回放必须有不同 action trace 和 trajectory hash；
- difference reward 为负或存在 safety cost 时，不具备晋级资格；
- 视频像素不参与 credit。

这解决了“Claw-10 拉走防守者但没有触球时是否应得分”的工程归因问题，同时禁止
critic 的预测值冒充真实反事实证据。

### 4. SimForge 任务面

新增两个 Soccer task spec：

- `soccer.continuous_match`：至少 60 秒、进球不终止、进球后继续比赛；
- `soccer.two_vs_one_decision`：至少 32 个种子、冻结低层、真实反事实贡献。

2v1 候选唯一允许修改 `/tactical_policy`，不会借战术训练覆盖 G1 小脑或肌肉记忆。

### 5. 官方环境探针

新增 `training/continuous_dm_control_probe.py`，它进行显式 goal-state fault injection，
目的只是验证官方环境的连续语义，不是制造球技成功：

- 2v2 BoxHead；
- 四个 agent；
- 60 秒；
- 2400 个 25 ms 控制步；
- 注入一次 goal detector 状态；
- 检查该步不是 terminal；
- 检查下一步球和位置已重开、goal detector 清除；
- 继续运行到 60 秒 time limit。

探针报告明确 `g1_policy_executed=false`、`agent_skill_claimed=false`、
`promotion_eligible=false`。

## 实验结果

### A. 60 秒连续 MuJoCo 环境合同

证据：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118-continuous-dm-control-v4/probe-report.json`

结果：

| 指标 | 结果 |
|---|---:|
| 物理时间 | 60.00000000001224 s |
| 控制步 | 2400 |
| agent 数 | 4 |
| goal reward | `[1, 1, -1, -1]` |
| goal 当步终止 | 否 |
| 场内重开后终止 | 否 |
| goal detector 清除 | 是 |
| 60 秒 time limit 终止 | 是 |
| 报告状态 | `PASS_ENVIRONMENT_CONTRACT` |
| 源提交 | `f521352a86f60fd01bb1dcd84f380ec885e25265` |
| 报告哈希 | `sha256:dee365fbbbd8b2c9ab202b421d7277b4645eb704423c48abebc1b61c655c200f` |

三份中间报告都没有被隐藏：v1 因 `get_pose()` 返回 MuJoCo 状态视图、未显式复制而
失效；v2 因验证器加固后实现哈希变化而失效；v3 虽通过物理门，但
`source_commit` 尚不包含未提交实现，也被主动降级。它们分别保存在带
`_invalidated_` 前缀的目录。最终 v4 显式复制每个快照、通过完整 gate-set 验证，且
源提交确实包含探针实现。

### B. 本地 G1 移动来球参考权重审计

输入为 HumanoidSoccer 官方 `policy_30000.onnx`，20 个固定种子试验，来球初速度
0.5–1.5 m/s，左右脚各 10 个，CPU MuJoCo 运行。原始证据：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118-humanoid-soccer-moving-ball-audit-v1/`

| 指标 | 结果 |
|---|---:|
| 触球 | 11/20 = 55% |
| 进球 | 1/20 = 5% |
| 左脚触球 | 6/10 |
| 右脚触球 | 5/10 |
| 已触球样本平均首次触球时间 | 2.281 s |
| 已触球样本平均方向误差 | 0.643 rad（约 36.9°） |
| 已触球样本平均最大球速 | 3.083 m/s |

原始文件哈希：

- `config.json`：`sha256:2eaeea74cef73759b6ee329a46c353caa778aeb00d7ab33cf616dfa8772c7301`
- `episodes.csv`：`sha256:ca3f0c1c14d3878063f54c0e4752f58b8a574fbaed6ba5d2461c13ca77fe4793`
- `summary.json`：`sha256:3fd5fad5c82b4f6d426ea9fafa94bd94f600a50610b98af56e8c117aef5cbc50`

结论不是“参考没用”，而是：它能提供左右脚、移动来球和自然运动先验，但在当前
更快来球分布上覆盖不足，并且任务目标仍偏向踢走/射门，不是把球处理到下一动作
最有利的位置。因此它只能进入 Teacher Portfolio，不能成为 First Touch Champion。

## ROSClaw 通用性边界

足球语义继续保留在 `rosclaw-soccer`：事件类型、第一脚失败、2v1 动作和角色奖励
都不进入 Core。Core 只复用已经合并的任务无关能力：

- `FailureConditionedDream`；
- `IndividualGrowthScope`；
- `FrozenPartnerSet`；
- `PlasticityLease`；
- `SuccessorStateContract`；
- `PairedDominanceEvidence`；
- `CanonicalChampionRegistry`。

只有当第二个非足球任务也需要相同的长回合/群体增长语义时，才把稳定交集提炼成
Core 的 `CollectiveGrowth`，避免再次把 G1 足球细节伪装成通用 ROSClaw 能力。

## 下一轮可直接执行的训练闭环

### S118-A：G1 First Touch

1. 从 PAiD、MotionDecode 足球片段和现有 G1 技能中建立只读 Teacher Portfolio；
2. 在 MuJoCo/MJX 中采样 0.5–6.0 m/s、左右 ±45° 来球和左右脚；
3. 特权教师读取真值球速/接触，学生只读有噪、延迟、遮挡后的观测和 1 秒历史；
4. 先 DAgger 覆盖，再用多 critic 约束 PPO 优化目标区、平衡和下一动作延迟；
5. 每个失败进入本文已实现的 Dream；
6. GPU 只做候选发现，最终在 CPU MuJoCo matched seeds strict replay；
7. 同时考旧射门、跑动和恢复 retention，候选不得通过遗忘换取接球成绩。

建议首个晋级门：128 acquisition + 128 retention，第一脚成功率至少提升 10 个百分点，
下一动作中位延迟不高于 0.70 秒，摔倒/越矩为零，旧技能成功率下降不超过 3 个百分点。

### S118-B：2v1“为什么传球”

1. 冻结通过 First Touch/传球/射门/恢复考核的低层 bundle；
2. 在 5–10 Hz 训练小型 recurrent tactical actor，集中式 critic 只用于训练；
3. 课程从 Defender 压持球人、封队友、左右偏置三类开始；
4. 每个候选对当前 Defender、历史 Defender、scripted Defender 做 matched exam；
5. 对无球跑位做真实焦点球员消融，使用 difference reward；
6. 只有 32+ holdout seeds 全部绑定、无低层漂移，才进入 Team Champion challenge。

### S118-C：G1 60 秒 Continuous Match

首版不是追求比分，而是证明以下连续链全部发生在同一时钟：

```text
接球 → 传/带 → 无球前插 → 射门 → 扑救/反弹 → 争抢 → 再组织
```

进球可以重开球，但不能重置 match clock、策略记忆、职业统计或失败归因。倒地允许，
但必须作为比赛内恢复事件继续；不能通过整段 reset 抹掉失败。

## 当前诚实边界

- 已实现并测试：连续比赛合同、First Touch 失败/Dream、2v1 反事实 credit、SimForge
  任务面、官方 60 秒 MuJoCo 环境 smoke。
- 已实测：本地 G1 移动来球参考权重在新分布上只有 55% 触球、5% 进球。
- 已实现：G1 物理 First Touch 采集器、单场景配对获取考试、独立确定性重放和
  证据下游 1080p 前后对照视频。
- 未实现：First Touch 上下文学生训练、2v1 learned tactical actor、历史 opponent
  pool、G1 60 秒连续比赛。
- 未宣称：任何真实机器人结果、First Touch Champion、团队战术成长或 2v1 成功率；
  当前 First Touch 只在一个低速中心右脚场景局部通过。

## 回归验证

- S118 定向测试：`12 passed, 3 skipped`；跳过项是尚未合并到当前 Core main 的三个
  extension registry 合同，不影响本轮本地对象测试。
- `mypy --strict src/rosclaw_soccer`：237 个源文件通过。
- 本轮 10 个 Python 改动文件：`ruff check`、`ruff format --check`、`compileall` 通过。
- 非 integration 全量：`717 passed, 14 skipped, 5 deselected, 11 failed`；干净
  `origin/main` 对照为 `709 passed, 14 skipped, 5 deselected, 11 failed`。11 个失败
  集合完全相同，均为 S78–S114 外部证据内容哈希/authority 已陈旧，不是 S118 回归。
- 全仓格式基线仍有 69 个旧文件不符合当前 formatter；本轮未进行无关的全仓机械改写。

本轮最大的进展不是又增加一个演示动作，而是把下一阶段的三个主张变成了可拒绝、
可追溯的合同，并用真实审计证明了为什么现有移动球权重还不足以支撑这些主张。
