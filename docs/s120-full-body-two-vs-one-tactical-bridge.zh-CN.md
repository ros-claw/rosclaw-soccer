# S120：从战术平面到三台全身 G1——2v1 学习决策物理桥

日期：2026-09-02

证据上限：`SIM_ONLY`

源码提交：`8ddf028d80c0396f9c52c0361ddaa9f2e62c92ad`

## 结论先行

S119 已证明高层 actor 能从物理成败中学会“什么时候传球”，但球员仍是战术平面上的
简化效应器。S120 去掉了这条边界：持球人、接球队友和防守者现在都是完整的 29-DoF
G1，三台机器人和一个球位于同一个 CPU MuJoCo 模型、同一个接触求解器中。高层 actor
仍然只能选择 `PASS / SHOOT`，不能直接写关节位置或力矩。

训练前封存的 8 个未见场景全部通过：

| 指标 | S120 封存结果 |
|---|---:|
| 三台全身 G1 / 共享球 / 共享求解器 | 是 |
| 学习 actor 安全成功 | **8/8 = 100%** |
| 与逐动作物理 oracle 一致 | **8/8 = 100%** |
| 位级独立精确重放 | **8/8 = 100%** |
| 动作覆盖 | PASS 4 / SHOOT 4 |
| 永远 PASS | 4/8 = 50% |
| 永远 SHOOT | 4/8 = 50% |
| 相对任一固定策略提升 | **+50 个百分点** |
| 平均 / 最大 regret | **0 / 0** |
| 最小已选 PASS 反事实贡献 | **+1.0** |
| 安全率 | **8/8 = 100%** |

因此本轮接受 `PASS_FULL_BODY_TACTICAL_BRIDGE`。它证明学习决策能驱动完整 G1 的
真实接触结果，而不再只是驱动平面圆点。它仍不是连续比赛、完整一脚传射或 Team
Champion；传球案例在队友脚边受控接触后终止，这个边界写入证据合同。

## 对 `rosclaw_soccer想法1.md` 的实质推进

文档建议从单项技能成长进入球员和球队成长，并把第一项可证伪任务定为
“2v1：学会什么时候传球”。S119 完成了学习与团队 credit 的最小闭环；S120 解决了
其最关键的待办：把 actor 接到真实全身技能执行器。

这一步推进了三层能力：

- **见天地**：球的运动、脚/身体接触、进球线、机器人支撑和跌倒都来自 MuJoCo
  状态，视频像素没有评分权；
- **见自己**：每台 G1 的 pelvis 高度、关节限位、力矩、饱和和身体接触都进入安全
  门；actor 与冻结 body、Athlete Foundation、First Touch 和动作技能内容哈希绑定；
- **见众生**：同一类持球状态下，防守者压迫持球人时队友的脚边接球产生 +1.0
  反事实贡献；防守者封传球侧时，actor 改为直接射门。

严格说，这仍是“具身社会情境认知”，不是自我意识。S120 的价值是把自己的身体、
队友和对手放进同一个可证伪的因果闭环。

## 系统架构

```text
冻结的全身低层能力
  ├─ G1 body / Athlete Foundation
  ├─ target-conditioned kick
  ├─ teammate stable locomotion stance
  └─ causal opponent locomotion
                │ 内容哈希绑定
                ▼
三台 29-DoF G1 + 一颗球 + 一个 CPU MuJoCo 求解器
                │
      每个状态穷举 PASS 与 SHOOT
                │
   正常轨迹 + 移除队友耦合的匹配反事实
                │
      失败重加权 ridge-Q tactical actor
                │
      训练前封存、训练不可见的 8 场考试
                │
所选动作 + 备选动作 + 反事实 + 独立精确重放
                │
    完整性验证器 + 证据下游 1080p 视频
```

### 1. 全身共享世界桥

新增 `training/full_body_tactical_2v1.py`。角色映射为：

- 持球人：无前缀 G1，执行冻结的 target-conditioned 全身踢球策略；
- 队友：`passer_` G1，保持冻结稳定站姿并提供真实脚/身体碰撞；
- 防守者：`goalkeeper_` G1 放到场内，使用冻结因果移动控制器追踪威胁；
- 球：标准共享自由球体，不复制、不传送、不按阶段改写轨迹。

共享世界增加了受边界校验的通用参数：角色开始时刻、动作目标、角色原点、门将威胁
来源和队友反事实碰撞开关。默认值保持原有三角色行为不变。

PASS 成功必须同时满足：持球人先真实触球、队友后真实触球、球心到左右脚踝的最近
距离不超过 0.28 m、防守者没有先截断。SHOOT 成功必须由真实触球后越过球门线，且
在队友或防守者先触球之前完成。

### 2. 身体安全不是动画观感

三个角色都检查有限状态和 pelvis 高度；同时检查 carrier/teammate/defender 关节
限位、全局力矩限位、驱动饱和与身体接触预算。正常足球对抗不能被粗暴地当成灾难，
因此安全合同允许最多 80 个 2 ms 接触步（0.16 秒上限），但绝不允许它掩盖跌倒、
关节或力矩违规。

本轮保留集的实际结果：

- PASS：持球人最低 pelvis 均为 0.6865 m，队友最低 0.7460 m；身体接触为 36–40
  个步，即 0.072–0.080 秒；
- SHOOT：持球人最低 pelvis 均为 0.6646 m，队友最低 0.7460 m，身体接触为 0；
- 全部案例无 joint-limit、torque-limit 或 actuator-saturation 违规。

### 3. 反事实不是让队友自由落体

最初的焦点队友消融把其所有碰撞类别清零，队友因此穿过地面自由落体，使仿真病态
变慢。这不是干净的团队 credit。最终实现使用独立碰撞类别：消融队友仍与地面接触、
保持数值支撑，但不能与球或其他球员耦合。

两次运行保持场景、种子、动作和冻结策略一致；轨迹还显式记录
`focal_teammate_present`。正常 PASS 能在队友脚边完成，消融后 PASS 奖励消失。封存
集中四次已选 PASS 的 difference reward 全部为 +1.0。

### 4. 数据学习和封存考试

采集集有 8 个状态，每个状态实际运行：

1. PASS 正常；
2. PASS 队友消融；
3. SHOOT 正常；
4. SHOOT 队友消融。

总计 32 次三机仿真，形成 16 份成对动作 evidence。沿用 S119 的失败重加权 ridge-Q
actor，只读取持球压力、队友线路开放度、射门线路开放度和进攻进度，不读取任何
“这是传球场景”的特权标签。训练集 8/8 分流正确。

保留集另有 8 个 scenario hash，训练前写入 `sealed-retention.json`，并声明
`training_access_allowed=false`。门限也在看到结果前固定为：任务成功、安全、动作
一致、精确重放均为 100%，PASS/SHOOT 各至少 4 次，必须同时击败两个固定策略，且
已选 PASS 的物理反事实贡献必须为正。

## 物理结果的通俗解释

四个“受压”案例里，防守者靠近持球人，队友在右脚传球策略当前能覆盖的接球走廊。
actor 选择 PASS。持球人在约 5.44 秒触球，球在 6.03–6.05 秒到达队友；球心到队友
脚踝为 0.148–0.176 m。相同状态盲目 SHOOT 会被队友身体挡住或失去任务奖励。

四个“封线”案例里，防守者靠近队友侧，当前低层策略若强行向弱侧传球会跌倒或被
截断。actor 选择 SHOOT；球在约 5.44 秒被踢出，6.36 秒越过球门线，交点为
`y=-0.848 m, z=0.241 m`。相同状态固定 PASS 失败。

所以 actor 学到的不是口号“多传球”，而是：**队友真正能接到时传；传球超出当前
身体能力域且射门通道存在时射。**

## 重要失败发现：低层技能仍不够宽

全身桥也揭示了平面实验看不到的限制。开发网格中，队友纵向偏离合格接球走廊约
2 cm 就可能错失接球，偏离约 4 cm 时部分布局甚至使持球人跌倒；当前 target-
conditioned kick 明显偏右脚，跨方向复用很弱。

因此 8/8 不能解释为“任意 2v1 已解决”。它只证明 actor 在已测全身 initiation set
内实现了正确选择。保留集在该集合内改变队友前后位置和防守布局，但没有覆盖任意
横向接球。这个失败域已保留，不能通过复制成功轨迹或扩大接球阈值解决。

下一次真正的低层突破应当是：用 First Touch、MotionDecode 模仿先验和持续 RL 扩大
左右脚、多角度、移动接球的可恢复集合，再重新封存更宽的 S121 retention。

## 代码、测试与证据

主要代码：

- `skills/team/shared_world.py`：受约束的角色位置/时序/目标和干净碰撞消融；
- `training/full_body_tactical_2v1.py`：三台全身 G1 的战术执行桥和物理评分；
- `training/full_body_tactical_growth.py`：采集、封存、考试、重放、持久化和完整验证；
- `media/full_body_tactical_video.py`：只读已通过证据的 1080p 严格重放；
- `tests/test_s120_full_body_tactical_growth.py`：SIM_ONLY、封存、几何状态、触球评分和
  反事实轨迹测试。

正式证据目录：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s120-full-body-two-vs-one-growth-v1/`

关键承诺：

- stage：`sha256:200770fb4dbef783d747ff2081a9e4dc46f052e6b94164c009c98717190194ec`
- actor：`sha256:08b55426bc0ff00915a3a16a74b269baaf6bdd7b16beceeaaf08aa27a841c824`
- training snapshot：`sha256:213efe813598b4576784b7126e3bce40f0b235bf7fe62de769d227d4be429f01`
- sealed retention：`sha256:a1572f5921804dc4a05b48bed0758f44f76b0a4532dff877bdaecd6df22abc25`
- retention report：`sha256:79398f639c52bbccd72498cd426609ad132a756e758d7bc6495be839f7e0fb91`

1080p 视频：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s120-full-body-two-vs-one-growth-v1/video/rosclaw-full-body-learned-2v1-1080p.mp4`

- 1920×1080，30 fps，558 帧，18.6 秒；
- 视频：`sha256:3083144da67ea494e5e9248008114869b4683cd07695bf2994d1643014085f4a`；
- manifest：`sha256:d79d9d9fed66804af1f415afca03fa5cbf439d62328d37151191283ece7fc039`；
- 明确标记 `visualization_only / SIM_ONLY / no pixel scoring`；
- RoboNaldo 外部资产许可证未证明允许商业宣传，因此 manifest 保持
  `commercial_use_allowed=false`。

验证结果：

- S120 单测：`3 passed`；
- S119 + S120：`10 passed`；
- 共享世界回归：`7 passed`；
- 新增/修改文件 `ruff`、`ruff format --check`、目标 `mypy`：通过；
- 全仓：`756 passed, 15 skipped, 11 failed`。11 个失败与 S119 时完全相同，均为已
  安装的 S78–S114 旧外部证据无法匹配当前实现哈希/closure，验证器按设计 fail-
  closed；S120 没有扩大失败集合。

## 下一阶段：从“选对”走向“连续做到”

S120 已完成 S119 的全身物理落地，但离 `想法1` 的连续球队还差四层：

1. **扩低层 initiation set**：左右脚、多角度移动接球、传后跑位和一脚出球；
2. **连续技能组合**：PASS 后不终止，接球队友完成 First Touch → 带球/射门，所有
   角色保持同一条时间轴和 episode identity；
3. **更完整动作空间**：加入 HOLD、左右 DRIBBLE 和无球跑位，使用时序 actor，而非
   单步 PASS/SHOOT；
4. **真正 opponent/team growth**：不同防守策略、角色独立记忆、失败回放、league
   和稳定性—可塑性门控，再进入 2v2/3v3 Continuous Match。

最优先不是再做一条更漂亮的固定镜头，而是扩宽“传球之后仍能站稳、移动、接住并
继续决策”的身体可恢复域。只有这样，S120 的战术智慧才会变成连续比赛能力。
