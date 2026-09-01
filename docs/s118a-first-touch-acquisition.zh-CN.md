# S118-A：G1 First Touch 局部获取闭环实施报告

日期：2026-09-01

实现分支：`codex/s118-continuous-soccer`

证据上限：`SIM_ONLY`

## 结论

本阶段第一次把“移动来球的第一脚处理”从讨论、失败枚举和静态合同推进到真实的
CPU MuJoCo 物理闭环：同一身体、同一来球、同一门限下，冻结基线因触球太重失败，
受限候选把 0.2 秒目标点误差从 `39.23 cm` 降到 `9.12 cm`，下一动作就绪时间从
`710 ms` 降到 `40 ms`，且没有关节越界、力矩越界或摔倒。候选在独立进程中的轨迹
和测量逐位一致，配对考试状态为 `PASS_PAIRED_ACQUISITION`。

这不是 First Touch Champion，也不是端到端神经小脑已经训练成功。当前结果只证明：

> ROSClaw Soccer 已经能从一个具体失败出发，形成受限候选，在真实 G1 MuJoCo
> 碰撞中测量改进，做独立回放，并由 fail-closed 配对考试拒绝或接受局部获取主张。

后续扰动检查也证实固定候选仍然脆弱：它不能可靠覆盖更快来球、左右横向偏移和
左右脚。因此没有写入 Champion Registry，没有覆盖旧策略，也没有取得任何真实
机器人权限。

## 与 `rosclaw_soccer想法1.md` 的关系

“想法1”要求项目从孤立动作升级为球员和球队的长期成长。本阶段刻意只推进路线表的
前两级，并为后续层级保留清晰权责：

| 层级 | 本阶段状态 | 证据边界 |
|---|---|---|
| Continuous Soccer | 已有 60 秒环境语义和不可拼接时间线合同 | 仅环境合同，不代表 G1 会连续比赛 |
| First Touch | 已有单场景 G1 物理获取闭环 | 局部通过，不晋级 |
| Dribbling | 未开始训练 | 无主张 |
| 1v1 attacker/defender | 已规划，未训练 | 无主张 |
| Off-ball movement | 未开始 | 无主张 |
| 2v1 tactical learning | 已有反事实贡献合同，未训练 actor | 无战术成功主张 |
| League / team dream | 仅架构方向 | 无实现主张 |

这样做避免把“某一脚踢得更准”包装成“已经会踢足球”，也避免让多智能体奖励掩盖
低层触球和身体稳定性的失败。

## 闭环架构

本阶段真实执行链为：

```text
移动来球场景
    ↓
冻结 RoboNaldo 全身先验 + 有界 First Touch 残差
    ↓
CPU MuJoCo G1 接触仿真
    ↓
原始 trajectory.npz
    ↓
解剖学首次接触 + 球速/方向/平衡/延迟测量
    ↓
稳定失败归因
    ↓
候选发现
    ↓
同场景 baseline/candidate 独立回放
    ↓
确定性 candidate replay
    ↓
fail-closed paired exam
    ↓
证据下游视频（像素不评分）
```

其中视频没有进入任何成功率、距离、速度或安全计算。所有数值都来自 MuJoCo 状态和
接触轨迹；视频只消费已经通过内容绑定的报告。

## 实现内容

### 1. 物理 First Touch runner

`src/rosclaw_soccer/training/first_touch_physics.py` 新增：

- 0.5–2.0 m/s 名义移动来球；
- 横向来球口袋、目标方向、目标出球速度和 0.2 秒测量窗口；
- 接球启动时序、站位、摆腿、COM、骨盆和脚姿态的有界候选参数；
- 来球驱动的 receiver phase sync；
- 触球前关节保护和触球后恢复接管；
- 原始轨迹、实现哈希、身体哈希、先验哈希和 Git 提交的内容绑定报告。

接触归因经过加固：第一处射手身体接触必须来自所要求的解剖学脚部；之后再出现脚部
接触，不能抹掉“身体先碰球”的因果事实。

### 2. 因果失败分类

`src/rosclaw_soccer/growth/first_touch.py` 区分：

```text
TOUCH_TOO_SOFT / MISS
WRONG_FOOT
LOST_BALANCE
TOUCH_WRONG_DIRECTION
TOUCH_TOO_HARD
TOO_SLOW_TO_NEXT_ACTION
```

未发生真实脚部接触时，不再同时声称方向错误或触球太重；方向、力度和选脚失败只在
存在相应解剖学接触后成立。平衡和下一动作延迟仍可独立失败。

### 3. Teacher Portfolio

`src/rosclaw_soccer/training/first_touch_teacher_portfolio.py` 对完整 MotionDecode 足球包
进行了可复现审计，而不是只挑几个好看片段：

| 项目 | 数值 |
|---|---:|
| CSV 片段 | 1,203 |
| 总帧数 | 1,867,958 |
| 总时长 | 15,566.32 s（约 4.324 h） |
| train / retention | 973 / 230 |
| Ball Control | 247 |
| Short Pass | 302 |
| Long Pass | 358 |
| Shooting | 152 |
| Others | 144 |
| 选定风格教师 | 24（控球/短传、左右脚平衡） |

三类教师权责被显式分开：

- MotionDecode：只提供姿态与协调风格，没有球状态和接触真值；
- PAiD：只作为移动球任务教师，不是 First Touch Champion；
- RoboNaldo：冻结的 G1 全身执行先验。

MotionDecode 和 PAiD 都有非商业研究限制，本轮 portfolio 与视频清单均明确
`commercial_use_allowed=false`。保留集指标在选教师时不可见，避免用 holdout 挑样本。

### 4. 配对获取考试

`src/rosclaw_soccer/training/first_touch_growth_exam.py` 要求：

- baseline 和 candidate 的场景、门限、身体、运动先验、实现和源码提交完全相同；
- 两者候选和物理轨迹必须不同；
- baseline 必须失败，candidate 必须通过且安全；
- candidate task loss 必须严格更低；
- candidate 必须由独立进程重跑；
- 重跑的场景、候选、provenance、测量、评价和轨迹哈希必须完全一致。

任一字段不满足就返回 `REJECTED_PAIRED_ACQUISITION`。单场景通过永远
`promotion_eligible=false`。

### 5. 证据下游视频

`src/rosclaw_soccer/media/first_touch_growth_video.py` 新增可复用渲染器：

- 只接受已通过且有确定性重放的配对考试；
- 验证 baseline、candidate 及轨迹文件的内容哈希；
- 同场景顺序播放 before/after，避免并排缩小导致接触不可见；
- 青色球表示触球后 0.2 秒的物理目标点；
- 1080p H.264，叠加结果与 SIM_ONLY 边界；
- 输出清单绑定视频和全部源证据；
- 无头命令会在 MuJoCo 上下文创建前自举到 EGL；
- 清单或任一源文件被修改后，验证器 fail closed。

## 局部配对实验

### 场景

| 参数 | 值 |
|---|---:|
| 名义来球速度 | 0.70 m/s |
| 触球前实测来球速度 | 约 0.391 m/s |
| 横向偏移 | 0 m |
| 目标方向 | -22° |
| 目标出球速度 | 2.20 m/s |
| 目标点时间 | 0.20 s |
| 使用脚 | 右脚 |

名义 0.70 m/s 的球在草坪摩擦下到达触球点时只有约 0.391 m/s，因此本场使用的
最小实测来球门为 0.30 m/s。这是早期低速课程，不是原计划 0.5–6.0 m/s 范围的完成
证明。

### 结果

| 指标 | 冻结基线 | 有界候选 | 变化 |
|---|---:|---:|---:|
| 0.2 s 目标误差 | 0.39226 m | 0.09118 m | -0.30108 m |
| 出球方向误差 | 1.356° | 0.162° | -1.194° |
| 下一动作延迟 | 0.710 s | 0.040 s | -0.670 s |
| 出球速度 | 3.7666 m/s | 2.4016 m/s | 从过重降到门内 |
| 最低骨盆高度 | 0.6758 m | 0.6475 m | 均过门 |
| 最大躯干倾角 | 20.76° | 26.08° | 均过门 |
| 最大根部速度 | 1.3373 m/s | 1.5496 m/s | 均过门 |
| 关节/力矩/摔倒 | 0 / 0 / 0 | 0 / 0 / 0 | 无安全回退 |
| task loss | 2.70948 | 0.32577 | -2.38371 |
| 状态 | FAIL | PASS | 局部获取通过 |

基线主要失败为 `TOUCH_TOO_HARD`，同时下一动作过慢。候选通过降低摆腿幅度、调整
COM/骨盆姿态并把接球时序推迟 0.20 秒，把球控制在目标口袋内。这里仍是有界参数
残差，不是神经网络直接输出 29 关节力矩。

### 确定性

候选由另一个进程完整重跑；以下对象逐位一致：

- measurement hash；
- evaluation hash；
- trajectory digest；
- 压缩轨迹文件 hash；
- provenance；
- report hash。

配对考试哈希：

`sha256:10c9c4a1954328c221e6044638437f807e8f3867023cd789edef6f1d55736edc`

## 泛化探针与失败分析

在不修改固定候选的开发探针中，加入名义速度 `0.65/0.70/0.75 m/s`、横向
`-5/0/+5 cm`、目标方向 `-15/-22/-30°` 和目标出球速度 `1.8–2.4 m/s`：

- 9 个上下文中通过 5 个；
- 中心来球的目标方向与目标速度变化较稳健；
- 0.65 m/s 出现漏球；
- 0.75 m/s 出现过重/错误方向或平衡失败；
- 左右横向偏移会漏球或导致身体失稳。

一个关键开发观察是：对 `y=-5 cm` 的来球，把站位残差从 `-6 cm` 改为 `-11 cm`
后，局部样本恢复到 `8.89 cm` 误差并安全通过。这证明上下文站位有作用，也同时证明
固定参数不能泛化。该观察只是开发探针，未进入正式配对证据。

### 左脚共享身体加固

左脚镜像探针进一步发现：关节动作已经做了矢状面镜像，但冻结右脚轨迹携带的初始
骨盆位置、全局四元数和初始关节状态此前没有一起进入左脚解剖坐标系；共享触球后恢复
也没有交换左右支撑脚语义。结果不是简单漏球，而是在恢复阶段摔倒。

提交 `4bea971` 在共享 G1 身体层修复了这两项：

- 左脚实例同时镜像初始根位置、根四元数和 29 关节姿态；
- 小脑恢复在 canonical 右脚坐标中运行，左右支撑状态交换，输出再镜像回物理左脚。

当前 CPU MuJoCo 左脚边界样本由 `pelvis≈0.07 m` 的倒地改善为：最低骨盆
`0.6771 m`、最大躯干倾角 `23.58°`、最大根部速度 `1.5646 m/s`，无关节/力矩
越界且不摔倒；独立重跑的测量和轨迹哈希完全相同。右脚通过样本的所有数值和轨迹
哈希保持不变。

左脚目前仍是 `TOUCH_TOO_SOFT`：它稳定地完成动作，但身体先进入来球口袋，要求的
左脚没有取得第一接触。因此这次只声称“修复双侧初始化与稳定性”，不声称“左脚
First Touch 通过”。边界证据位于：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118a-first-touch-left-boundary-v1/`

物理响应还呈现明显非线性：相邻摆腿幅度可能从安全控球跳变为漏球或摔倒。因此下一步
不能靠手写 if/else 为每个速度打补丁；需要把失败邻域变成数据，由上下文 actor 学习
站位、触球时序和接触强度，同时让安全 critic 和 retention 门约束更新。

## 数据驱动下一步

下一轮 S118-A2 应执行：

1. 建立速度、横向、方向、左右脚平衡的 acquisition/retention split；
2. 用 MotionDecode 只做全身姿态/自然度编码，用 PAiD 只做移动球任务 proposal；
3. 由 MuJoCo 真值生成脚部接触、0.2 秒目标和安全标签；
4. 先训练低权限 contextual residual actor，输出站位、时序和接触强度，不直接拿到
   硬件或 Champion 权限；
5. 使用 DAgger 补齐 actor 在横向来球和速度边界上的错误状态；
6. 再用 constrained actor-critic 优化任务损失，安全 critic 独立否决；
7. 每一类失败进入 `FailureConditionedDream`，但 Dream 样本与正式 holdout 隔离；
8. CPU MuJoCo 对 128 acquisition + 128 retention 做 matched strict replay；
9. 左右脚、速度和横向分层均达标且旧射门/恢复下降不超过门限后，才允许 challenge
   Champion；
10. First Touch Champion 产生后，冻结低层进入 2v1 “为什么传球”的高层训练。

建议下一轮优先攻克顺序：

```text
中心低速右脚
→ 横向 ±5 cm 右脚
→ 0.5–1.0 m/s 右脚
→ 镜像左脚
→ 带噪观测学生
→ 下一动作 PASS/DRIBBLE/SHOOT
→ retention + Champion challenge
```

## 证据位置

- Teacher Portfolio：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118a-first-touch-teacher-portfolio-v1.json`
- baseline/candidate/replay 与配对考试：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118a-first-touch-paired-v1/`
- 1080p 视频：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118a-first-touch-growth-video-v1/rosclaw-first-touch-growth-1080p.mp4`
- 视频清单：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s118a-first-touch-growth-video-v1/rosclaw-first-touch-growth-1080p.json`

视频参数：`1920×1080 @ 30 fps`、`402` 帧、`13.4 s`、H.264；视频 SHA-256 为
`750827f8c4e03dc8f5400f2260dd2e65c30c8f5c4ec05c07ca0c2b902308dd02`。

## 代码与验证

阶段提交：

- `8ebe239 feat: close first-touch physics acquisition loop`
- `40b70ff test: require deterministic first-touch replay`
- `9b29ae9 feat: render evidence-bound first-touch growth`
- `4bea971 fix: stabilize bilateral kick initialization`

已完成：

- S118/S118-A 聚焦测试：18 个通过；
- 新增媒体模块 `ruff`、format、compileall、strict mypy 通过；
- 无 `MUJOCO_GL` 环境变量的独立 720p CLI 回放通过；
- 1080p 成片由 manifest 二次验证通过；
- 全量非 integration 回归在干净 Core main 上为 `731 passed, 14 skipped,
  5 deselected, 11 failed`。

11 个全量失败与本分支新增代码无关，都是 S78–S116 外部历史证据在当前实现下的
内容/实现哈希漂移；聚焦新增测试无失败。全仓仍有历史 formatter 基线差异，本阶段未
机械改写无关文件。

## 最终边界

- 已证明：单个低速、中心、右脚 First Touch 场景的可重复局部净改进；
- 已证明：完整 MotionDecode 足球数据可以被审计、隔离 train/retention 并按许可分权；
- 已证明：失败、候选、物理回放、配对考试和视频可形成一条内容绑定链；
- 未证明：左右脚、横向和高速来球泛化；
- 未实现：端到端神经小脑、在线 actor-critic 持续更新、直接关节力矩策略；
- 未证明：First Touch Champion、2v1 战术、G1 连续比赛或真实机器人效果；
- 未发送：任何 ROS、DDS、硬件或电机命令。
