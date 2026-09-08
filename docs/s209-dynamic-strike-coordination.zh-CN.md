# S209：数据驱动的连续接球—快速射门协调

## 结论先行

S209 打通了一个可审计的最小学习闭环：MuJoCo 失败/成功轨迹不再只用于人工调参，而是经过
逐字节校验后组成数据快照，拟合出带数据集哈希的运行时 actor，再由独立 CPU MuJoCo 考场和
精确重放验证。固定考场中，相对当前父基线：

| 指标 | 当前父基线 | S209 learned actor | 变化 |
|---|---:|---:|---:|
| 接球到出脚 | 4.70 s | 4.38 s | -0.32 s（-6.8%） |
| 出脚后峰值球速 | 7.14 m/s | 10.88 m/s | +3.74 m/s（+52.3%） |
| 无阻挡弹道目标误差 | 2.381 m | 1.302 m | -1.079 m（-45.3%） |
| 完整阶段链 | 1→2→3→4→5→6 | 1→2→3→4→5→6 | 保持 |
| 世界安全 | 通过 | 通过 | 保持 |
| 精确重放 | 通过 | 通过 | 保持 |

这不是“进球成功”。学习策略的射门在球门前被蓝方 playmaker 的左小腿真实封堵：接触位置
`x=4.779 m`、接触力 `459.8 N`、球速变化 `4.35 m/s`。无阻挡弹道在门框内，但碰撞后的
真实门线交点在门框外。因此本阶段只认证：

`DATA_BOUND_QUICK_STRIKE_WITH_PHYSICAL_OPPONENT_SHIN_BLOCK`

不认证进球、门将扑救、故意封堵、跨目标泛化或真实机器人迁移。

## 为什么这次不是动作补丁

S208 的接球后流程由 CAPTURE、ORIENT、PLANT、STRIKE、RECOVER、COMPLETE 六个显式阶段
组成。S209 没有给关节追加一段脚本，也没有改写机器人根节点或足球状态，而是在 ORIENT 阶段
加入一个受限策略接口。策略只能输出两个无量纲混合量：

1. `stance_blend`：在随球移动目标与预测支撑站位之间混合，范围 `[0, 0.80]`；
2. `goal_yaw_blend`：在来球朝向与射门线朝向之间混合，范围 `[0, 1.0]`。

输入来自六项在线身体/任务观测：阶段进度、支撑深度、横向误差、支撑朝向误差、接近朝向误差
和球速。策略没有关节、力矩、根节点、足球、阶段切换或真实硬件权限；其激活上限硬编码为
`SIM_ONLY`。

参数教师只负责发现可行轨迹，`policy_type=parameter`，不能宣称为模型。学习流程验证教师报告
哈希、NPZ 文件哈希、轨迹摘要、actor 哈希，并逐帧重新计算教师动作；只有安全、完整、稳定恢复
且无阻挡弹道整球在门框内的教师轨迹可进入正样本数据集。拟合后策略必须为
`policy_type=learned_linear`，并绑定不可为空的 `dataset_snapshot_hash`。

## 闭环数据流

```text
CPU MuJoCo 参数教师
        │
        ├─ 成功轨迹 ── 哈希/动作重算/阶段门 ──┐
        └─ 失败轨迹 ── 可序列化失败记忆       │
                                               ▼
                                  immutable dataset.npz
                                               │
                                               ▼
                           归一化双头 linear-sigmoid actor
                                               │
                                               ▼
                          独立 primary + replay CPU MuJoCo
                                               │
                         ┌─────────────────────┴─────────────┐
                         ▼                                   ▼
                 固定考场证据通过                    跨目标泛化门拒绝
```

本轮生成了 111 份探索性 probe 报告。它们包括完整射门、未进入射门、关节边界拒绝、射偏、
门将接触以及场上球员封堵等结果；最终报告不引用未密封的探索统计，只引用最终教师、学习产物、
primary/replay 和父基线。

## 新增与修改模块

- `growth/dynamic_strike_coordination.py`
  - 受限观测、动作与 NumPy actor；
  - 参数常量/阶段斜坡教师；
  - learned actor 强制数据集承诺和 SIM_ONLY 边界。
- `skills/team/independent_team_world.py`
  - 在 ORIENT 阶段接入受限 actor；
  - 保存逐帧观测、动作和激活状态；
  - actor 缺席时保留父路径；观测切片先复制，避免未来出现 MuJoCo 可写 view 风险。
- `training/dynamic_strike_coordination_probe.py`
  - 单候选物理探针；成功与失败都保存 NPZ；
  - 使用出脚后 120 ms 内最大正向速度做独立弹道评估；
  - 验证器重新计算报告、阶段、弹道、actor 与轨迹哈希。
- `training/dynamic_strike_coordination_learning.py`
  - 校验教师轨迹并构造不可变数据快照；
  - 拟合归一化双头 actor；
  - 加载时重新验证 dataset/actor/report 三层承诺。
- `training/dynamic_strike_coordination_exam.py`
  - 将父基线、学习 artifact、primary/replay 和真实对抗接触密封；
  - 区分“无阻挡弹道射正”与“碰撞后真实门线结果”；
  - 固定考场可以通过，但明确禁止自动晋升为通用策略。
- `media/dynamic_strike_coordination_video.py`
  - 同片展示父基线与 learned actor；
  - 视频只消费已通过的物理报告，像素不参与打分。
- `growth/phase_conditioned_strike_assessment.py`
  - 失败指标由 JSON 不支持的 `Inf` 改为 `null`，使失败轨迹可以可靠进入记忆和训练闭环。

## 证据与视频

- 密封考场：
  `/code/rosclaw/phase8_evidence/s209-dynamic-strike-exam-v2/dynamic-strike-coordination-exam.json`
- 学习数据与 actor：
  `/code/rosclaw/phase8_evidence/s209-dynamic-strike-distilled-v3/`
- learned primary/replay：
  `/code/rosclaw/phase8_evidence/s209-dynamic-strike-distilled-eval-v3/`
- 1920×1080、30 fps、24.43 秒对比视频：
  `/code/rosclaw/phase8_evidence/s209-dynamic-strike-exam-v2/s209-learned-quick-strike-comparison.mp4`
- 视频 manifest：
  `/code/rosclaw/phase8_evidence/s209-dynamic-strike-exam-v2/s209-learned-quick-strike-comparison.json`

视频中的慢放、机位和球尾迹只是可视化；物理结论全部来自 NPZ、MuJoCo 接触和密封报告。

## 重要失败与解释

### 1. 更猛烈并不等于更好

部分阶段斜坡把接球到出脚压到约 4.5 秒，但触发关节边界、丢失 STRIKE 阶段或把球踢出门框。
因此动作上限不是性能目标，失败轨迹必须保留。

### 2. 任务变化会改变完整队伍动力学

改变足球初始横向位置后，有的场景安全但没有形成传球接收，有的场景直接触发身体边界。改变
球门目标位置也会影响全队决策、站位和后续碰撞；固定参数响应高度不平滑。

### 3. 第一版多目标 learned actor 没有泛化

用目标横向 `0.6/0.7/0.8 m` 的三个成功教师拟合线性策略后，独立评估只在部分场景安全完成，
且并非全部弹道在门框内。该候选没有晋升。最终 S209 actor 是对固定考场教师的保守蒸馏，不能
外推为通用射门小脑。

### 4. 当前父基线在本机数值环境下被旧门控拒绝

父基线仍能同进程精确重放，并完成传球、接球、射门、门将接触和恢复；但当前环境记录到门将
手套 `111.5 N` 与身体 `293.9 N` 同帧接触，未满足冻结的“手套力必须主导”条件。因此报告
保留 `parent_passed_under_current_environment=false`，没有篡改阈值来让它变绿。

## 下一阶段（S210）

1. 在观测中加入显式目标上下文和防守者到射门线的距离，避免同一身体误差对应多个任务；
2. 用保守 mixture-of-experts 或 kernel memory 表达不连续可行域，保留冻结父专家解决
   stability-plasticity；
3. 用 DAgger 式闭环把 learned rollout 的分布偏移状态交回安全教师标注；
4. 新增“真实越过门线 / 合法门将扑救 / 合法场上封堵”三分支 outcome gate；
5. 在 `0.6–0.9 m` 目标带、不同来球速度和不同防守站位上做留出测试，只有全部安全且优于父策略
   才允许通用晋升；
6. 将该协调策略接口抽象到 ROSClaw Core 的通用 Growth candidate/evidence contract，足球特征、
   场景和媒体仍留在 Soccer 仓库。

当前最重要的工程结论不是“已经成为球星”，而是：ROSClaw Soccer 首次把连续动作调参、失败
保存、数据快照、模型拟合、独立物理重放、真实对抗接触和拒绝泛化串成了可验证闭环。
