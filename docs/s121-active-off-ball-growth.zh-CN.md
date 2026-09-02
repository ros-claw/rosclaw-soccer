# S121：三台 G1 全员跑起来——无球跑位与动态防守 Growth 闭环

日期：2026-09-02  
证据上限：`SIM_ONLY`  
正式物理源码提交：`49c1a6a335e0a22e53790849ebbd31e35c0ffa84`

## 结论

S120 证明了全身 G1 能学会在 2v1 中选 `PASS / SHOOT`，但接球队友平均 86.0% 的时间
接近静止，防守者平均 73.14% 的时间接近静止。用户指出“像木头人”是正确的：旧循环
虽然给三个角色都加载了神经 locomotion policy，却把非持球人的速度命令持续设为零。

S121 修复了这个架构缺口，并完成一次失败驱动的选优—封存考试闭环。冻结战术 actor
仍只选择传球或射门；新的战术移动层给队友生成 `CHECK / RUN` 目标，给防守者生成
`PRESS / COVER` 目标，再由每台 G1 自己的冻结 RoboNaldo 神经步态执行全身关节动作。
没有脚本改写机器人 root pose、关节或球状态。

正式未见保留集结果：

| 指标 | S120 | S121 |
|---|---:|---:|
| 任务成功 / 安全 | 8/8 / 8/8 | **8/8 / 8/8** |
| 独立精确重放 | 8/8 | **8/8** |
| 动作覆盖 | PASS 4 / SHOOT 4 | **PASS 4 / SHOOT 4** |
| 队友平均位移 | 0.0446 m | **1.3762 m** |
| 队友接近静止比例 | 86.0% | **0.286%** |
| 队友主动运动比例 | 5.43% | **95.71%** |
| 防守者平均位移 | 0.3843 m | **0.7201 m** |
| 防守者接近静止比例 | 73.14% | **2.14%** |
| 防守者主动运动比例 | 14.89% | **55.43%** |
| 队友每局左右摆腿切换 | 未设门 | **平均 24 次** |
| 防守者每局左右摆腿切换 | 未设门 | **平均 21 次** |
| 动作质量联合门 | 未设门 | **8/8** |

最终状态为 `PASS_ACTIVE_OFF_BALL_GROWTH`。这证明三台机器人可以在保持原有传射结果
和身体安全的同时持续活动，不再是一个球员踢球、两个人形路桩围观。

## 实现了什么

### 1. 通用 SIM-only 战术移动合同

`skills/team/shared_world.py` 新增：

- `G1MovementWaypoint`：世界坐标、单调仿真时钟上的不可变地面目标；
- `G1TacticalMovementConfig`：最大速度、最大加速度、位置增益和到达半径；
- `_command_tactical_movement`：目标插值、位置误差反馈、速度/加速度投影、世界坐标到
  角色局部坐标转换，再输入冻结的神经 locomotion policy；
- 轨迹记录实际目标、实际速度命令和活动标志。

该接口永久 `SIM_ONLY`、`hardware_authorized=false`。它不允许写关节、root pose 或球速，
因此“跑起来”仍要经过 G1 29-DoF 动力学、脚地接触、策略输出和力矩限制。

这不是只为足球写的关节补丁：任何共享 MuJoCo G1 角色都可以复用同一 waypoint →
学习型 locomotion 执行接口。足球只提供本轮的目标生成和考试任务。

### 2. 动作条件化角色 option

`FullBodyRoleMovementPlan` 将高层战术动作编译成多角色 option：

- PASS：队友从斜后侧做 check-and-receive 切入，在约 6.03 秒进入真实脚部接球窗口；
- SHOOT：队友做纵深支援跑，避免挡住持球人的射门线；
- PASS 防守：对手做 pressure arc，但不能粗暴横穿球路制造必然拦截；
- SHOOT 防守：对手移动覆盖传球侧，保留当前 actor 判定的射门通道。

移动目标目前是有界动作模板，身体执行来自神经策略。它比“站桩”有实质进步，但尚未
达到根据每一帧队友、对手和球状态自主生成路线的多智能体策略；这一边界没有隐藏。

### 3. 不只看位移的动作质量门

单纯要求“移动一米”会奖励滑行、抽搐或大步冲过任务点。S121 联合检查：

- carrier / teammate / defender 位移；
- 小于 0.05 m/s 的停滞比例和大于 0.15 m/s 的主动比例；
- 左右脚水平摆动的交替切换；
- 腰臂等上身关节实际活动范围；
- 战术速度命令的峰值加速度；
- 原有 pelvis 高度、倾角、关节限位、力矩、饱和、身体接触和任务结果。

封存集队友上身平均关节活动范围为 0.1815 rad，防守者为 0.1447 rad；并非把刚体 root
在地面上平移。当前 root speed jerk 仍偏大，视频中仍能看到小步频和策略本身的机械感，
所以下一阶段不能把“通过动作门”宣传成“已经像顶级职业球员”。

## Growth 如何从失败中选出方案

训练前先写入 8 个不可见保留场景。训练仅在另 8 个 DEVELOPMENT 场景比较三种候选：

| 候选 | 任务成功 | 动作质量 | 联合通过 | 结论 |
|---|---:|---:|---:|---|
| compact | 8/8 | 4/8 | 4/8 | 动得太保守，射门支援可以，传球跑位质量不足 |
| athletic | **8/8** | **8/8** | **8/8** | 选中 |
| aggressive | 4/8 | 8/8 | 4/8 | 看起来更积极，但冲坏 4 个真实传球 |

这个结果很重要：Growth 没有选择动作幅度最大的 aggressive，而是选择同时保持任务、
安全和动作质量的 athletic。大动作不能洗掉足球失败。

开发中还保留了两次 fail-closed 记录：

1. 第一版防守者直接穿过球路，在 5.48 秒先触球，传球失败；通过物理结果调整为压迫弧线；
2. 第一版封存布局让冻结 actor 输出未支持的 `HOLD`，考试在消费该物理场景前停止；
   修复支持域并从零重跑，而不是把 HOLD 偷偷映射为 PASS；
3. 首次完整封存物理 8/8 通过，但报告把枚举 wire value `pass/shoot` 与大写字面量比较，
   最终状态仍为 REJECTED。增加回归测试后再次从零生成正式证据。

失败证据分别保存在：

- `s121-active-off-ball-growth-v1-failed-hold-support/`
- `s121-active-off-ball-growth-v1-rejected-wire-value/`

## 正式证据与视频

正式目录：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s121-active-off-ball-growth-v1/`

关键承诺：

- stage：`sha256:babcb402ea798b22ebe04f13d2699908e24c19ecce34423d75d85fd2d8e69e45`
- retention：`sha256:70f9de347a9fd2abae8faaaffa0b17ef78a67ffd4177a0f7b6d2b9968d20c4e2`
- selected route：`sha256:860136c4d7e84e4c9fe4dbd9960811e98620db08426105d09c65028f9be24be0`
- sealed manifest：`sha256:ae8ce203c9d0fde36a49c73f83f5a839b82112fdffd9a1cb9d40e1b8541a626c`

独立完整性验证：`VALIDATED_ACTIVE_OFF_BALL_STAGE`，零错误。

46.13 秒、1920×1080、30 fps 视频：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s121-active-off-ball-growth-v1/video/rosclaw-active-three-g1-growth-1080p.mp4`

- 4 个不同封存案例：两个 PASS、两个 SHOOT；
- 每例包含全程宽镜头和接触链慢回放；
- video hash：`sha256:0af097413b2e585b3a62382bbdeaeea8bf23aa305502a180e36745f004822637`；
- manifest hash：`sha256:65ac6c187808736ad14336f722acb4cc920dcf74ad108760a7d184fa69b965d9`；
- 只作证据下游可视化，像素不参与评分；
- 外部 RoboNaldo 资产商业宣传许可未证明，因此 manifest 明确
  `commercial_use_allowed=false`。

代码验证：

- S120 + S121 单测：`7 passed`；
- 共享世界相关选择集：`25 passed`，另 2 个旧外部证据因实现哈希变化按设计失效；
- 以内容绑定的 S117 Core 环境跑全仓：`760 passed, 15 skipped, 11 failed`；11 个失败与
  S120 完全相同，均为 S78–S114 已安装外部证据与当前实现/closure 哈希不一致，没有
  扩大失败集合；
- 当前 ROSClaw main 已不含历史 `simforge.reproducibility`，直接使用 main 收集全仓会有
  6 个 S111–S114 import error，因此没有把旧模块复制回当前 Core 来制造假绿；
- 本轮文件 `ruff`、`ruff format --check`、目标 `mypy`、`compileall` 通过。

## 离“真正顶级球员”还差什么

这一轮解决的是“全员从站桩变成有任务的连续步态”，不是终点。视频仍可看出步幅偏小、
跑动速度低、转身和观察缺乏、root 速度 jerk 较高、接球队友触球后没有继续组织进攻。
下一阶段应按以下顺序推进：

1. 用球、队友、对手和自身本体状态训练 5–10 Hz 的 recurrent multi-agent route actor，
   取代固定时间 waypoint 模板；
2. 用 MotionDecode / 足球运动数据作风格教师，对当前物理合格步态做 tracking-RL / AMP
   蒸馏，奖励跨步、转髋、摆臂、注视和预备姿态，而不是直接回放动作；
3. 训练 First Touch → DRIBBLE / PASS / SHOOT 连续 option，让传球在接住后继续同一 episode；
4. 引入竞争性 defender league 与角色独立 actor-critic，让进攻与防守共同提高，而不是
   让防守者按已知动作走固定弧线；
5. 扩展左右脚、转身接球、变速、假动作和 2v2，再在封存对手与扰动上考试。

真正的突破门应是：路线由观测闭环产生、动作风格来自数据但任务来自物理、三角色都能
在失败回放后持续更新，并且新能力不破坏 S120/S121 的任务、安全和精确重放基线。
