# S119：学会“为什么传球”——封存物理 2v1 战术成长闭环

日期：2026-09-02

证据上限：`SIM_ONLY`

实现分支：`codex/s119-learned-2v1`

## 结论先行

本轮完成了 `rosclaw_soccer想法1.md` 中第一个可证伪的团队智能实验：低层 G1
Athlete Foundation、First Touch、传球和射门身份全部冻结，只让一个 10 Hz 高层 actor
学习 `PASS / SHOOT`。它不是按固定阈值写死的补丁，而是从 32 个 CPU MuJoCo 物理状态、
64 份配对动作证据中拟合动作价值，再到训练前封存、训练不可见的 24 个新状态中考试。

最终结果：

| 封存考试指标 | 结果 |
|---|---:|
| 未见场景 | 24 |
| 学习策略任务成功 | **24/24 = 100%** |
| 永远传球 | 16/24 = 66.7% |
| 永远射门 | 16/24 = 66.7% |
| 相对两种固定策略净提升 | **+33.3 个百分点** |
| 与逐动作物理 oracle 一致 | 22/24 = 91.7% |
| 安全/有限状态 | 24/24 = 100% |
| 独立精确重放 | 24/24 = 100% |
| 平均 regret | 0.04182 |
| 最大单例 regret | 0.50184（门限 0.55） |
| 最小已选传球反事实贡献 | 0.81763 |
| 动作分布 | 传球 10，射门 14 |

这证明的是一个**边界明确的 L2 社会智能能力**：防守者封队友时学会射门，防守者
压持球人时学会传球；并且学习策略在保留集上显著优于两个固定动作。它还不是三台
完整 G1 在同一个世界中的 2v1，也没有产生 Team Champion。当前物理学习世界是位于
全身控制器上方的战术平面，使用冻结技能效应器表示已通过的低层能力。这一限制写入
证据、模型和视频合同，不能被下游工具抹掉。

## 对“见天地、见自己、见众生”的推进

本轮主要推进“见众生”，但没有绕过前两层：

- **见天地**：球、草地摩擦、角色接触、拦截、接球和越过球门线都来自 CPU MuJoCo
  状态；视频像素不参与评分。
- **见自己**：actor 绑定 Athlete Foundation 和整个低层技能 bundle 的内容哈希；
  任意低层能力改变都会拒绝执行，而不是把身体退化隐藏在团队得分后面。
- **见众生**：相同持球状态下，Defender 选择压迫持球人还是封堵队友会改变传球/射门
  的实际物理回报；actor 因此开始根据队友、对手和空间关系做决策。

这还不是“自我意识”。更准确地说，它是 ROSClaw 可持续成长架构中的一个社会情境
感知与因果 credit 原型。

## 闭环架构

```text
冻结 G1 低层 bundle
        │ 内容哈希绑定；训练无修改权限
        ▼
CPU MuJoCo 2v1 战术平面
        │ 每个状态都实际跑 PASS 和 SHOOT
        ▼
配对动作价值 + 焦点球员物理消融
        │ 初态/种子/环境相同，轨迹必须不同
        ▼
失败重加权 ridge-Q actor
        │ 只在 10 Hz 输出 PASS/SHOOT；OOD → HOLD
        ▼
训练前封存的 24 场保留集
        │ 未见横向站位、压迫程度、反应延迟、速度、摩擦
        ▼
选中动作 + 未选动作 + 消融 + 独立 replay
        │ 所有门同时通过才接受 bounded retention
        ▼
内容绑定证据与下游分析视频
```

### 1. 冻结低层能力

`FrozenTacticalSkillBundle` 同时绑定：

- G1 body profile；
- Athlete Foundation；
- First Touch actor；
- 传球技能；
- 射门技能。

actor 保存并核验 foundation 和 bundle 哈希，`direct_joint_torque_output=false`。因此
这次训练没有权限调整关节、力矩、小脑或肌肉记忆，也不能通过牺牲身体稳定性提高
战术成绩。

本轮还从提交后的源码重新跑了一次既有三角色 G1 共享世界开发证据，用于确认冻结
技能身份不是空字符串：

- 传球触球：是；
- 传球落点误差：0.023479 m；
- 射门触球和进球：是；
- 射门目标误差：0.077882 m；
- 有限状态：是；
- joint/torque violation：无；
- strict replay：是。

该旧开发合同的顶层 `passed=false / REJECTED_DEVELOPMENT` 是刻意的非晋级语义；其
内部共享世界结果通过，作用仅是重新确认本轮冻结低层身份，不是借此晋级团队能力。

### 2. 连续物理 2v1 决策面

新增 `training/tactical_2v1_physics.py`。每场最长 4 秒，MuJoCo 步长 2 ms，技能效应器
50 Hz，高层决策 10 Hz。场上存在一个真实自由球体、持球人/接应者/Defender 的碰撞
体和接触效应器。

Defender 的 `commitment` 是连续量：

- 接近 0：封堵队友和传球线路，直接射门更合理；
- 接近 1：压迫持球人和射门线路，传给空位队友更合理；
- 中间区域：结合线路开放度、压力和进攻进度决定。

成功必须来自球与角色的物理接触、接球或越过球门线。代码没有给某个 commitment
直接标注“正确动作”。Defender 反应延迟、速度、球地摩擦也进入物理仿真和保留集。

### 3. 真反事实贡献，而不是 critic 自说自话

对每个动作都运行两次独立物理轨迹：

1. 正常存在接应者；
2. 相同初态、种子、环境和动作，但从模型构建时移除接应者的触球能力。

两条轨迹必须有不同 action trace 和 trajectory hash，difference reward 来自实际物理
结果差，不接受 critic 估计冒充。保留集中，被 actor 选择的 10 次传球，其最小正向
反事实贡献仍为 0.81763。这说明至少在这个平面内，队友对成功确实产生了物理贡献。

### 4. 数据驱动高层 actor

新增 `growth/tactical_2v1_actor.py`。输入为五个归一化状态量及其交互特征：

- 持球人压力；
- 队友线路开放度；
- 射门线路开放度；
- 球到球门的进度；
- 队友到球门的进度。

32 个 acquisition 状态分别实际跑 `PASS` 和 `SHOOT`，得到 64 个物理动作价值。actor
先做一次 ridge-Q 拟合，再把预测错误或动作价值边界接近的状态作为 hard examples 以
4 倍权重重放。这一轮不是在线 actor-critic；它是数据驱动、失败重加权的高层离线
策略，目的是先把团队 credit、封存考试和遗忘边界做对。

actor 还具备三条 fail-closed 约束：

- 低层 bundle 哈希变化：拒绝；
- 超出训练支持域：输出 `HOLD`；
- 模型文件、训练样本、实现哈希被篡改：加载或证据验证失败。

### 5. 先封存，再训练，再独立重放

24 个 retention 状态在训练开始前写入 `sealed-retention.json`，并显式声明
`training_access_allowed=false`。acquisition ledger 记录训练过程未读取 retention
manifest。保留集与训练集的 scenario hash 完全不重合，并改变：

- 队友左右站位与未见横向距离；
- Defender commitment；
- Defender 反应延迟；
- Defender 速度；
- 草地摩擦。

每个保留状态不仅执行 actor 选中的动作，还执行另一个动作作为物理 oracle；然后对
选中动作做焦点球员消融和单独精确重放。晋级门要求任务成功、动作覆盖、oracle 一致
率、regret、安全、重放和正向传球贡献同时满足。

## 实验中的失败与成长

第一次稀疏采集只覆盖“明显压迫”和“明显封堵”。actor 在中间 commitment 边界上会
选错：训练点正确，但稍微改变 Defender 位置就错误切换动作。这不是通过改保留集或
降低成功门限解决的，而是把边界附近的 0.48–0.72 commitment 加回 acquisition，
并对低 margin/错误状态做 hard-example 重加权。

最终 24 场中仍有 2 场与逐动作物理 oracle 不一致。两场所选动作都成功、安全、严格
重放，整体最大 regret 0.50184，低于事先门限 0.55。因此本轮接受的是
`PASS_BOUNDED_TACTICAL_RETENTION`，不是“最优战术”或 Champion。视频中传球案例的
置信度只有 0.36，也如实显示；后续需要更多对手分布和时序记忆，而不是掩盖不确定性。

## 代码与测试

主要新增：

- `src/rosclaw_soccer/training/tactical_2v1_physics.py`：MuJoCo 2v1 物理环境、匹配消融；
- `src/rosclaw_soccer/growth/tactical_2v1_actor.py`：内容绑定 actor、OOD fallback；
- `src/rosclaw_soccer/training/tactical_2v1_growth.py`：采集、封存、考试和完整验证器；
- `src/rosclaw_soccer/media/tactical_2v1_growth_video.py`：证据下游 1080p 分析视频；
- `tests/test_s119_two_vs_one_growth.py`：因果、学习、篡改、封存、重放、视频测试。

测试覆盖的关键负例包括：

- 不安全、未封存或修改过的 retention 被拒绝；
- Defender commitment 改变时物理优选动作确实翻转；
- 接应者消融后传球 difference reward 必须下降；
- OOD 状态只能 `HOLD`；
- 修改 actor、轨迹、stage 或视频来源时验证失败；
- 保存的 primary/replay 轨迹语义摘要必须完全一致。

## 证据与视频

正式成长证据：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s119-two-vs-one-tactical-growth-v2/`

关键承诺：

- stage：`sha256:9727679e6307880fb7b8c1d59a18339ff3c7bf95f7e6ee85fddfadaa81761f7a`
- actor：`sha256:fb64607ad8734a7d02477673fc61225190ea3b1ffa6713e368583febc2e5041f`
- training snapshot：`sha256:3065a87ceb20996e391cfb4c02bc25c391e52b07bb53bed184c0efa1b3fe0b76`
- sealed retention：`sha256:fdf25aaa3b6077659f574ab069742d1830f25f5b47d6249a422c56908608437c`
- retention report：`sha256:1085f808c5b2f8cf60ce4be55fe49c666a5723a44e3181cc7951ac5babcb7465`
- source commit：`f404e2d3670391cd58c31876539ec88352d75830`

冻结 G1 技能复核：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s119-frozen-g1-skill-bundle-v1/`

1080p 分析视频：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s119-two-vs-one-tactical-growth-v2/video/rosclaw-learned-2v1-1080p.mp4`

- 分辨率：1920×1080；
- 时长：12.6 秒；
- 视频哈希：`sha256:38e48e36fd5c5d4dcbf7d0c3e7ef498cd0d25a4366b20c74e0621fb793691431`；
- manifest：`sha256:8d12f519403ec930bd030bd409c6d03bcd0f887b9d0a44eefef903cc80c9854d`。

视频展示两种未见场景中的学习决策和最终对照指标。它明确标注
`TACTICAL PLANE / g1_bodies_rendered=false / visualization_only=true`，不是全身 G1
宣传片，也不具备商业使用或晋级资格。

首个 v1 证据是在完整 stage 验证器进入源码提交之前生成，已主动保留并改名为
`s119-two-vs-one-tactical-growth-v1_invalidated_validator_added`，没有拿旧证据冒充最终
结果。v2 由包含验证器的已提交源码重新生成并通过独立进程验证。

## 最终验证

- S118 + S119 定向：`14 passed`；其中 S119 为 `7 passed`，覆盖因果翻转、反事实、
  actor/OOD、封存证据、篡改拒绝和视频绑定；
- `mypy --strict src/rosclaw_soccer`：249 个源文件通过；
- S119 五个 Python 文件：`ruff check` 与 `ruff format --check` 通过；
- `python -m compileall -q src tests`：通过；
- stage 完整验证器：`PASS_BOUNDED_TACTICAL_RETENTION`；
- 视频 manifest 与 `ffprobe`：H.264、1920×1080、30 fps、378 帧、12.6 秒，全部通过；
- 全仓：`753 passed, 15 skipped, 11 failed`。对照上一阶段是
  `746 passed, 15 skipped, 11 failed`，新增 7 项全部通过，失败集合没有扩大。11 个旧
  失败均为 S78–S114 已安装外部证据与当前实现哈希或 reproducibility closure 不再一致，
  验证器按设计 fail-closed；不是 S119 行为断言失败。

Soccer 的完整测试需要堆叠包含 SimForge reproducibility 和 continual contracts 的
ROSClaw Core 源码。只叠入用户当前旧 Core 工作目录会在收集期缺模块；最终全仓数字
使用干净的 `rosclaw_core_s117` 提交工作树。这个依赖条件被保留在报告中，没有把收集
失败解释成测试通过。

## 尚未完成与下一阶段

本轮完成了“为什么传球”的最小数据闭环，但 `想法1` 中更大的球队成长仍有明显缺口：

1. 用三台完整 G1 的共享世界执行器替换战术碰撞效应器，把同一个 actor 接入真实
   First Touch、跑位、传球、射门和恢复；
2. 加入 `HOLD / DRIBBLE_LEFT / DRIBBLE_RIGHT`，并使用 recurrent actor 处理 Defender
   假动作和时序遮挡；
3. 建立 scripted/current/historical Defender pool，防止只会对一种防守者；
4. 训练无球队员主动拉开、前插和二过一，并继续用焦点球员消融做 contribution credit；
5. 在同一 60 秒比赛时钟跑通接、传、跑、射、扑、反弹和恢复，而不是若干短回合拼接；
6. 完成上述物理桥接后，再考虑多 GPU recurrent actor-critic 和 Match Dream。GPU 只做
   候选发现，晋级仍回到 CPU MuJoCo 封存种子、严格重放和旧技能 retention。

下一阶段最重要的不是扩大宣传口径，而是消除本轮的 `tactical_plane_only`：让学习到
的传射选择真正驱动完整 G1 技能链，同时保持现在已经建立的封存、反事实、内容绑定
和 fail-closed 证据纪律。
