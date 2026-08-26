# S94 球队交替 Growth 与阶段能力账本

日期：2026-08-26
状态：`READY_FOR_ALTERNATING_GROWTH`（SIM_ONLY）

## 1. 本轮结论

本轮不是又调一组门将参数，而是修正球队学习架构：正式禁止传球手、射手、门将三个候选在同一轮同时改变。今后每一轮严格执行：

```text
1 名球员 PLASTIC
+ 2 名队友 FROZEN
+ 相同场景 / 相同环境 / 相同 seed
+ discovery 与 sealed holdout 分离
```

早期 `role_learning.py` 的“三角色原子联合晋级”仍保留为历史基线，但不再作为讨论 4 与“尝试突破”路线的推荐入口。本轮新增的 `alternating_team_growth.py` 才是球队后续成长合同。

## 2. 为什么这不是普通的胜负奖励

一场比赛被切成六个物理阶段：

```text
LEAD_PASS
→ RUNNING_INTERCEPT
→ STRIKE
→ GLOVE_SAVE
→ CONTROLLED_LANDING
→ SUCCESSOR_READY
```

每段分别记录：

- 是否成功；
- 当前动作质量；
- 交给下一动作时的 `successor_value`；
- 安全代价；
- 起止物理时间；
- 执行策略、源证据和轨迹哈希；
- 是否严格重放；
- 是否使用视频像素评分（必须为 false）。

这样，门将扑救率提高不能由“射手变差”冒充；射手进了一个漂亮球，也不能掩盖射后失稳；起身站高了但不能接下一次横移，同样不能晋级。

## 3. 交替晋级门控

候选若要替换某一角色，必须同时满足：

1. 只有该角色的 artifact 和 generation 改变；
2. 候选严格绑定当前 parent，generation 只加一；
3. 两名冻结队友的完整 binding 逐字段不变；
4. 可塑角色所拥有的每一个阶段均不能降低成功率；
5. 阶段质量平均提升至少 `0.01`；
6. successor value 平均提升至少 `0.005`；
7. 冻结队友的阶段质量和 successor 回归不得超过 `0.01`；
8. 整链成功率不退化；
9. 所有阶段 safety cost 为零；
10. 同一个候选还要在完全不相交的 sealed holdout 上再次通过。

任何一个条件不满足，candidate 可以进入 Failure Memory，但不会进入球队 roster。

## 4. Failure-conditioned Dream

新增角色私有失败记录和确定性 Dream 扩展。不同失败不会再统一扔回 PPO：

| 失败 | Dream 变量 |
|---|---|
| 传到跑动者身后 | 接球队员速度、朝向、到达时间 |
| 触球前停住 | 接近速度、传球速度、拦截角 |
| 射门偏离目标区 | 目标高度、横向目标、触球相位 |
| 手套漏球 | 冲击时刻、横向、高度 |
| 落地不安全 | 根角动量、摩擦、冲量 |
| 二次扑救未 ready | 第二威胁间隔、方向、恢复摩擦 |

Dream 只允许使用 discovery failure，明确拒绝 sealed holdout 泄漏。

## 5. 用 S93 v4 真实证据建立球队账本

输入是已经通过当前实现哈希校验的 S93 v4：

- 源证据 SHA-256：`a51b2d265985b8b0edaad0d07871f73f3793d607706484534431f451e71f23c4`；
- 两个标准球门左右死角案例；
- 每个案例各有无门将反事实轨迹和有门将扑救轨迹；
- CPU MuJoCo 严格重放；
- 没有用视频像素评分。

S94 将两条完整链切成 12 个阶段记录。代表性阶段分数如下：

| 阶段 | 左侧 | 证据含义 |
|---|---:|---|
| 传球 quality | 0.907 | 传球误差约 4.63 mm |
| 传球 successor | 0.925 | 横向交接误差约 2.25 mm |
| 跑动拦截 quality | 0.553 | 从传球到射门触球约 1.788 s |
| 射门 quality | 0.320 | 横向贴柱但高度只约 1.249 m |
| 手套扑救 quality | 0.572 | 实体表面接触且扑出 |
| 落地 quality | 0.505 | 落地受控，但仍有继续优化空间 |
| recovery-ready quality | 0.9998 | 尾窗直立性很高，速度很低 |

这些分数是当前合同下的归一化工程指标，不是跨项目排行榜。

## 6. 账本暴露出的真实能力边界

已经通过、应保留为历史锚点：

- 固定接应点精确传球：2/2；
- 约 1.249 m 高度的横向贴柱射门：2/2；
- 基于可见球角度预站位后的双侧贴柱扑救：2/2；
- 单次扑救后的稳定恢复尾窗：2/2。

仍未测试，不能宣称已经学会：

- 给运动中队友的真正提前量传球；
- 贴近 2.44 m 横梁的上死角射门；
- 门将从中央临场覆盖标准球门上死角；
- 扑救、恢复后再扑第二个物理来球。

因此新调度器没有把 2/2 成功继续当作主要训练数据，而把它们降为 `HISTORICAL_ANCHOR`；四个未测难题进入 `PROBE_UNTESTED`。

## 7. 下一轮严格顺序

```text
Round 1: Playmaker PLASTIC
         Finisher + Goalkeeper FROZEN
         任务：dynamic lead pass

Round 2: Finisher PLASTIC
         Playmaker + Goalkeeper FROZEN
         任务：upper dead-corner strike

Round 3: Goalkeeper PLASTIC
         Playmaker + Finisher FROZEN
         任务：center-origin upper-corner save

Round 4: Goalkeeper PLASTIC
         Playmaker + Finisher FROZEN
         任务：save → recover → second threat
```

该顺序是依赖顺序，不是四个候选一起更新。每轮只有前一角色通过 individual gate 和 team compatibility gate 后，才冻结成下一轮对手/队友。

## 8. 证据

- 机器可读账本：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s94-team-growth-ledger-v1/team-growth-ledger.json`
- 内部 `ledger_hash`：`sha256:1f0acfd628e613d0fdc03d3fffd5f33cecd7dabc1214e6b90e8a728407a5467a`
- 文件 SHA-256：`01a6084e64e630baecb8f8f5ab75b6afcccc43adfec7d9b078c484c585361094`
- 源演示视频仍为 S93 v4：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s93-regulation-dead-corner-save-v4/regulation-dead-corner-showcase.mp4`

本轮没有把旧视频冒充新技能视频，账本中明确记录 `fresh_training_performed=false`。下一段新视频必须来自 Round 1 的 fresh physics episodes。

## 9. 验证

- S94 针对性测试：7 passed；
- S84/S85/S89/S90/S92/S93/S94 联合回归：37 passed；
- 全仓回归：610 passed、15 skipped、3 failed；三个失败仅为 S78/S79/S80 冻结证据的旧 `implementation_hash` 被当前实现按设计拒绝，未伪造重签；
- 全仓 Ruff：通过；
- 新增三个文件 Ruff：通过；
- 新增三个文件 mypy：通过；
- compileall：通过；
- S93 v4 源证据当前实现、请求、四条轨迹哈希复验：通过；
- S94 账本篡改（把一轮可塑角色从 1 改成 3）：验证器拒绝。

全仓 `ruff format --check` 仍报告 72 个历史文件需要格式化；没有为了形式上的绿灯批量改写这些证据绑定源文件，否则 S78–S93 的实现哈希会继续失效。全仓 mypy 也仍包含既有 recovery/MJX 第三方 stub 与两处 `recovery_reachability.py` 类型问题；本轮新增四个源/测试入口的 mypy 已独立通过。

证据上限仍为 `SIM_ONLY / CPU_MUJOCO / non-commercial research`，没有发送 ROS、DDS、Unitree SDK 或真实硬件命令。
