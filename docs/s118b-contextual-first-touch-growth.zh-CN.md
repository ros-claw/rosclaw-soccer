# S118-B：上下文 First Touch 数据驱动成长与接触闭环报告

日期：2026-09-02

实现分支：`codex/s118-continuous-soccer`

证据上限：`SIM_ONLY`

## 结论

本轮把 S118-A 的“单场景手工有界候选”推进成了第一条可训练、可拒绝、可封存保留集、
可独立复演的数据驱动 First Touch 闭环。最重要的同场景配对结果是：冻结接触模式把球
踢得过重，`0.2 s` 目标误差为 `20.98 cm`；学习到的踝部接触残差把误差降到
`8.05 cm`、下一动作等待从 `620 ms` 降到 `40 ms`，且 G1 没有摔倒或越过安全门。
候选在独立 CPU MuJoCo 进程中逐位复现，配对考试为 `PASS_PAIRED_ACQUISITION`。

在训练完成后才公开的两个新目标条件中，失败感知原型 actor 均通过：目标误差分别为
`7.74 cm` 和 `7.35 cm`，独立复演 `2/2` 完全一致。这证明 actor 在两个已经验证的
来球接触口袋内，可以把接触肌肉记忆迁移到未见过的目标方向和目标出球速度。

这仍不是 First Touch Champion。横向来球口袋之间的连续插值仍为 `0/2`，左脚还没有
取得合格的解剖学“要求脚先触球”，高权限在线拦截反射虽然能让右脚先碰球，却把出球
速度推得过高而 `0/2` 未通过任务门。因此本轮没有覆盖旧策略、没有晋级 Champion，
也没有发送任何 ROS、DDS、硬件或电机命令。

## 为什么这轮不是“又写了几个踢球补丁”

S118-A 已经证明一个有界候选能改善单一场景，但它本身不能回答：换一个来球或目标，
应该使用哪一种肌肉记忆？本轮把该问题拆成四个可复用层：

```text
MuJoCo 本体/接触真值
    ↓
稠密接触诊断（谁先碰球、相差多少、脚离球多远）
    ↓
内容绑定训练快照（成功 + 失败经验）
    ↓
低权限 contextual actor / failure-aware prototype actor
    ↓
有界全身候选 + 六维局部接触残差
    ↓
既有力矩/稳定性安全投影
    ↓
独立 CPU 物理回放 + acquisition/retention 门
    ↓
接受局部肌肉记忆，或把失败重新送回 Growth
```

actor 的输入是来球速度、横向位置、目标方向、目标速度和用脚语义；输出是受限站位、
触球帧、摆腿幅度、COM/骨盆和接触关节残差。它没有真实机器人权限，也不能绕开
共享世界中的力矩、根部速度、骨盆高度和恢复安全门。

## 代码实现

### 1. 稠密物理接触诊断

`training/first_touch_physics.py` 新增逐步接触诊断：

- 第一处 G1 身体—球接触的几何体、时间和顺序；
- 要求脚的第一次接触以及与身体接触的时间差；
- 要求脚到球的最小距离、发生时刻和距离变化趋势；
- 诊断只从 MuJoCo 状态与碰撞读取，视频像素不参与评分；
- 场景、候选、拦截配置、实现、身体、先验和轨迹均进入内容哈希。

该诊断让 `body-first` 不再被后续脚部接触掩盖，也把“差一点踢到”和“动作方向完全
错误”区分开来，为失败驱动训练提供了比单个成功/失败标签更密集的监督。

### 2. 六维局部接触残差

`FirstTouchCandidate` 增加有界、平滑的六维关节接触残差，覆盖髋俯仰、髋滚转、膝、
踝俯仰、踝滚转和躯干/支撑协调窗口。残差只能在限定触球窗口内生效，并继续经过既有
全身稳定器和安全投影。

在 `y=-3 cm` 的移动来球场景中，训练发现踝俯仰 `+0.10 rad` 的局部接触记忆能把
过重触球转成受控 First Touch；另一个来球口袋学习到了膝和踝组合。这些值被保存为
内容绑定 prototype，不再通过场景 ID 或手写分支选择。

### 3. 两类上下文 actor

`growth/first_touch_context_actor.py` 实现了受限 ridge actor：

- 只消费经报告和轨迹哈希验证的训练样本；
- 标准化上下文，预测有界 residual；
- 超出训练域时回退，不外推；
- 保存特征、范围、权重、训练快照与 actor 哈希；
- 加载时重新计算哈希，篡改即拒绝。

实验发现，线性参数插值在接触动力学中并不安全：两个端点都可以脚先触球，中间值却
可能让躯干先接触球。为此新增 `growth/first_touch_prototype_actor.py`：

- 保留完整成功接触模式，不对接触几何做盲目平均；
- 用成功/失败记忆共同决定最近的安全 contact mode；
- 对未覆盖上下文拒绝或回退；
- 让来球横向位置/速度负责“怎么接触”，目标方向/速度只调整接触模式内的任务参数；
- 模型、训练证据和每次决策均内容绑定。

这不是宣称 prototype 永远优于神经网络，而是当前数据稀疏且碰撞拓扑不连续时，先用
可审查的离散接触模式避免危险插值；后续数据足够后，再在模式内部训练连续 actor。

### 4. 本体感拦截反射

`growth/first_touch_interception.py` 和 `skills/team/shared_world.py` 增加实验性闭环：

1. 从 MuJoCo 读取球—要求脚的相对位置与速度；
2. 用实测脚部 Jacobian 将任务空间修正力投影为关节力矩；
3. 限制增益、力和关节力矩；
4. 与原控制叠加后再次经过共享安全投影；
5. 在 trace 中记录是否激活、误差、力和力矩。

低权限反射对接触顺序作用不足；提高权限后 `2/2` 能使右脚先接触球且骨盆保持约
`0.66 m`，但出球达到 `4.35–4.82 m/s`，因此任务门 `0/2`。这个控制器仍保留为
实验能力，但不会被训练程序或 actor 自动启用，更不会因“脚碰到了球”就获得晋级。

## 实验闭环与失败记录

### A. Ridge actor：训练成功不等于物理保留成功

初始训练集包含 `8` 个物理报告（`4` 个通过、`4` 个失败），覆盖中心和横向来球。
ridge actor 能拟合受限参数，却在封存测试上 `0/2`：G1 身体比要求脚早约 `40 ms`
触球。原因不是数值拟合误差，而是接触拓扑发生了切换。

结论：接触控制不能只看参数 MSE；必须把“解剖学谁先触球”作为独立物理门。

### B. Prototype actor：盲目最近邻仍不够

首次 prototype 仍在横向插值 holdout 上 `0/2`，同样是 `body-first`。随后加入失败记忆
并重标定上下文尺度，明确区分两类权责：

- 来球横向和速度决定接触口袋；
- 目标方向和目标速度在已验证口袋内决定任务条件。

初版尺度曾让目标方向压过来球位置，造成 `1/2`；修正后新的、训练完成后封存的目标
条件达到 `2/2`。失败证据没有删除，旧 retention seal 也被显式标记为已打开、不得
继续用来主张泛化。

### C. 同场景配对 Growth

场景为右脚接 `y=-3 cm`、名义 `0.70 m/s` 的移动来球。两条轨迹使用相同场景、身体、
运动先验、安全门、实现和源提交，只有接触候选不同。

| 指标 | 冻结接触模式 | 学习接触残差 | 改善 |
|---|---:|---:|---:|
| `0.2 s` 目标误差 | 0.20977 m | 0.08048 m | -0.12929 m |
| 出球方向误差 | 8.095° | 7.288° | -0.806° |
| 出球速度 | 2.91847 m/s | 2.27464 m/s | 更接近目标口袋 |
| 下一动作延迟 | 0.620 s | 0.040 s | -0.580 s |
| 最低骨盆高度 | 0.63990 m | 0.64492 m | 均通过 |
| task loss | 2.05719 | 0.65151 | -1.40568 |
| 任务状态 | `TOUCH_TOO_HARD` | `PASS` | 局部获取通过 |

候选独立复演的 measurement、evaluation、trajectory digest、压缩轨迹文件和 report
哈希完全一致。配对考试哈希为：

`sha256:9c9b25847281fa864aaab0708ddf64e646ba88b414a729f3ba02b33174213a92`

### D. 新目标条件封存保留集

保留集在 actor 定型后封存，禁止训练访问。它只改变目标方向/目标速度，不跨越已验证
的来球接触口袋：

| 场景 | 来球横向 | 新目标方向/速度 | 目标误差 | 方向误差 | 下一动作 | 结果 |
|---|---:|---:|---:|---:|---:|---:|
| center-target | 0 cm | -20° / 2.28 m/s | 0.07743 m | 2.162° | 0.040 s | PASS + 精确复演 |
| latneg-target | -5 cm | -24° / 2.28 m/s | 0.07347 m | 1.235° | 0.040 s | PASS + 精确复演 |

两例均为要求的右脚先接触球，最低骨盆分别为 `0.64746 m`、`0.63843 m`；primary 与
独立 replay 在场景、候选、测量、评价、报告和轨迹哈希上全部一致。阶段摘要自哈希为：

`sha256:a26e6d5220d1a7c152f400c0b733740cc8eb1e0b710f3dc517e6b4fed911589e`

## 宣传视频与使用边界

同场景 before/after 视频：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118g-first-touch-residual-paired-v1/video/rosclaw-first-touch-residual-growth-1080p.mp4`

参数：`1920×1080`、`30 fps`、`402` 帧、`13.4 s`、H.264。视频哈希：

`sha256:4f034e0932fab82212a3e06fd414364d30d7803e1371b4310777cb944443ee55`

视频只消费已经通过的配对报告和物理轨迹；像素不参与评分。由于本阶段仍使用含
非商业研究约束的运动先验链，manifest 明确记录 `commercial_use_allowed=false`；
可以内部展示和研究讨论，但不能据此声称商业可用资产或真实机器人效果。

## 证据位置

- actor、封存保留集与阶段摘要：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118f-first-touch-prototype-actor-v3/`
- 同场景 baseline/candidate/replay 与配对考试：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118g-first-touch-residual-paired-v1/`
- 两个保留场景及独立复演：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118f-retention-center-target-v1/`
  和 `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118f-retention-latneg-target-v1/`
- 视频及内容绑定 manifest：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118g-first-touch-residual-paired-v1/video/`

## 下一阶段

下一轮不应扩大宣传主张，而应集中攻克本轮明确暴露的三个物理瓶颈：

1. 在每个已验证 contact mode 内训练连续 actor，用 DAgger 收集脚—球最小距离和
   body-first 边界状态，禁止跨接触拓扑直接插值；
2. 把拦截反射拆成“低权限捕获 + 学习到的卸力接触”，同时优化脚先触球和出球速度，
   而不是单纯提高 Jacobian 力；
3. 单独建立左脚 acquisition 课程，先取得稳定的解剖学左脚第一接触，再谈双脚共享；
4. 扩展来球速度、横向和入射角分层，每层 acquisition/retention 严格隔离；
5. GPU/MJX 只负责大批量候选发现，最终主张仍由独立 CPU MuJoCo matched replay；
6. First Touch 真正晋级后，冻结该低层 bundle，进入 2v1 接—传—跑连续 Growth，避免
   高层战术训练反向破坏新学到的接触肌肉记忆。

## 代码质量与回归

- S118 定向测试：`33 passed`；
- 全量测试：`746 passed, 15 skipped, 11 failed`；
- `ruff check .`：通过；
- 本分支相对 `origin/main` 的 25 个 Python 变更文件：format check 通过；
- `mypy --strict src/rosclaw_soccer`：245 个源文件通过；
- `python -m compileall -q src tests`：通过。

全量的 11 个失败与进入本轮前已记录的集合一致，均为 S78–S116 已安装外部历史证据
在当前实现下的内容/实现哈希漂移；fail-closed 验证器正确拒绝了陈旧证据。它们不是
本轮新增回归，也没有被重签或绕过。全仓 format check 仍有 68 个历史旧文件不符合
当前 formatter，本轮没有机械改写无关代码。

## 最终边界

- 已证明：一个失败场景通过局部接触残差获得安全、确定性净改进；
- 已证明：两个已知来球接触口袋内，对未见目标条件的右脚保留为 `2/2`；
- 已实现：稠密接触诊断、内容绑定 actor、失败记忆、封存保留集、实验性在线拦截反射；
- 未证明：横向口袋之间连续泛化、不同来球速度泛化、左脚 First Touch；
- 未实现：端到端神经小脑 Champion、持续在线 actor-critic 晋级、直接关节力矩策略；
- 未授权：任何真实机器人动作或 ROS/DDS/硬件执行。
