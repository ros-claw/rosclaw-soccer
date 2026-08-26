# S51：失败驱动恢复教师桥与固定路由留出验证

日期：2026-08-20
状态：**训练桥闭环通过；不可晋级、不可部署、仅限仿真**

## 1. 先说结论

S49 已经证明，旧 R0 起身专家在自己的仰卧扰动分布上可以 16/16，但放到门将真实扑救后
的 9 个稳定终态上是 **0/9**。本轮没有继续给 R0 调阈值，而是按“恢复基础能力”重建了
一条失败驱动的教师桥：

```text
真实 post-save 快照
  → 本体姿态/关节状态匹配官方多姿态起身参考
  → 官方 OpenTrack specialist2 参考跟踪专家
  → 物理结果选择 teacher motion / entry phase
  → 必要时降低参考相位速度
  → 捕获 upright ready pose
  → 冻结参考并连续稳定 2 秒
```

正式开发矩阵共运行 63 个 CPU MuJoCo 物理试验：

- 官方教师自身参考邻域：**9/9**；
- post-save 候选迁移：9 个快照 × 3 个入口候选 × 2 个时间伸缩，共 **54** 次；
- 最近入口、原速直接使用：**7/9**；
- 允许失败驱动地换入口、仍保持原速：**8/9**；
- 最后一个困难状态使用已经预先声明的 2× 时间伸缩：**9/9**；
- 所有选中路线均达到 **连续 2 秒稳定**。

随后冻结全部 9 条开发路线，再生成结果未知的局部扰动留出集：

- 9 个基准快照 × 3 个扰动 = **27** 个未见状态；
- 路由重选：**0**；
- 通过：**27/27（100%）**；
- 每一个基准状态：**3/3**；
- 95% Wilson 下界：**87.54%**；
- 有限数值状态：**27/27**。

这说明教师桥不只是记住 9 组精确浮点数。但它仍然是**带参考相位的特权教师**，而且
运行在与 post-save 数据源不完全等价的 OpenTrack MuJoCo 场景中，所以本轮明确返回
`promotion_eligible=false`。

## 2. 为什么这是 S49 的关键修正

旧 R0 的源动作是 supine 起身；S49 的真实终态却是：

| 姿态 | 数量 |
|---|---:|
| PRONE | 8 |
| LEFT_SIDE | 1 |
| SUPINE | 0 |

所以 0/9 不是偶然，也不是把骨盆阈值放宽一点能修好，而是**训练分布与前序技能终态分布
断裂**。本轮采用的官方 specialist2 是在 6 条 `fallAndGetUp` 动作上训练 20 亿步的
29DoF G1 策略，包含多种倒地与起身相位，第一次给真实 post-save 分布提供了相符的动作
拓扑。

更重要的是，匹配器不只找“长得像的倒地帧”。候选入口必须同时满足：

1. 当前帧确实是低骨盆、非直立状态；
2. 该参考在限定未来窗口内确实存在连续直立 successor；
3. 按身体坐标系重力、29 关节 RMSE、骨盆高度共同排序；
4. 同一动作中相近相位做非极大抑制，避免三个候选其实是同一帧附近的复制品。

选择规则也不允许“相似度好看”压过真实物理结果：

```text
successor-state success
  > 更少时间干预
  > 更低峰值根角速度
  > 更短执行时间
  > 状态匹配分数
```

## 3. 正式选中的 9 条恢复路线

| S49 env | 原姿态 | 官方动作 | 入口帧 | 相位伸缩 | 到稳定时间 | 峰值根角速度 |
|---:|---|---|---:|---:|---:|---:|
| 9 | PRONE | fallAndGetUp2_subject2 | 1250 | 1× | 18.62 s | 7.47 rad/s |
| 52 | PRONE | fallAndGetUp2_subject2 | 7324 | 1× | 18.62 s | 6.90 rad/s |
| 48 | PRONE | fallAndGetUp2_subject2 | 1360 | 2× | 35.24 s | 12.14 rad/s |
| 14 | PRONE | fallAndGetUp2_subject2 | 7348 | 1× | 18.62 s | 8.14 rad/s |
| 44 | PRONE | fallAndGetUp2_subject2 | 6232 | 1× | 18.62 s | 6.62 rad/s |
| 13 | LEFT_SIDE | fallAndGetUp2_subject2 | 7348 | 1× | 18.64 s | 10.33 rad/s |
| 22 | PRONE | fallAndGetUp2_subject2 | 6230 | 1× | 18.62 s | 7.35 rad/s |
| 30 | PRONE | fallAndGetUp2_subject2 | 1246 | 1× | 18.62 s | 7.53 rad/s |
| 10 | PRONE | fallAndGetUp2_subject2 | 1404 | 1× | 18.62 s | 12.69 rad/s |

这里的 18.62 秒是完整参考恢复到 ready 并再连续稳定 2 秒的执行时间，不是面向最终
比赛的延迟目标。它证明了“存在可行恢复路径”，下一阶段仍需通过学生蒸馏、相位压缩和
successor-state 奖励把恢复时间降下来。

## 4. 固定路由扰动留出集

留出集在开发路线冻结后才由内容哈希确定，每个原快照生成 3 个样本。它同时扰动：

| 量 | 均匀扰动硬上限 |
|---|---:|
| 29 关节位置 | ±0.020 rad |
| 29 关节速度 | ±0.050 rad/s |
| 根部 roll/pitch | 每轴 ±0.015 rad |
| 根部线速度 | 每轴 ±0.030 m/s |
| 根部角速度 | 每轴 ±0.050 rad/s |

随机种子绑定：

```text
seed_namespace
+ perturbation_config_hash
+ base_snapshot_hash
+ sample_index
```

因此可以精确复现，且留出考试不能根据结果换动作、换帧或换时间伸缩。验收合同在运行前
冻结为：总通过率至少 80%，每个基准状态至少 2/3，全部状态有限。实际为 27/27，满足
合同；但最大根角速度达到 **14.35 rad/s**，说明“能恢复”已经打通，“自然、快速、安全”
仍未完成。

## 5. 本轮开发内容

### ROSClaw Core：通用能力

足球只负责暴露问题，以下能力放在 core，不含 G1、球门或门将词汇：

- `SkillSuccessorState`：当前技能的终态必须进入下一技能可接管的连续状态包络；
- `SuccessorStateGrowthObjective`：把下一技能价值纳入当前技能优化目标；
- `GrowthSafetyProfile`：明确分开 SIM_ONLY 探索安全与严格晋级安全；
- `FailureConditionedDream`：围绕精确失败快照按内容绑定的扰动分布出题；
- `CapabilityFrontierScheduler`：优先训练约 30%–70% 成功的能力边缘；
- `PlasticityLease`：多智能体训练中只允许一个焦点策略变化，其余策略哈希必须冻结。

这些合同同样适用于“抓取失败→manipulation-ready”“导航碰撞→navigation-ready”，不是
足球专用补丁。

### ROSClaw Soccer：领域实现

- `recovery_foundation.py`：Absorb/Get-Up/Athlete 三专家、本体 gate、soft blend 和
  `GOALKEEPER_READY` 合同；
- `recovery_reference_catalog.py`：外部论文/项目/身体/许可证/用途的固定目录；
- `recovery_baseline_adapter.py`：HumanUP、HoST 不同动作语义的显式 23→29DoF 适配；
- `recovery_teacher_bridge.py`：安全 NPZ、successor-aware 入口挖掘、物理优先路线选择、
  确定性扰动留出；
- `opentrack_recovery_bridge_exam.py`：配对开发矩阵、场景审计、原子报告、fsync 试验日志；
- `opentrack_recovery_bridge_holdout.py`：开发报告验签、固定路由、独立扰动考试和统计下界。

所有 NPZ 都使用 `allow_pickle=False`，策略与配置、动作源、快照 corpus、场景和试验日志
都由 SHA-256 绑定。运行中断后只复用绑定完全一致且单条哈希验证通过的 trial。

## 6. 外部参考与可复现绑定

已下载到 `/code/rosclaw/rosclaw_football/repos`：

| 项目 | 固定 commit | 本轮用途 |
|---|---|---|
| HumanUP | `7516e0f27e6f` | 两阶段起身探索/约束基线 |
| HoST | `70bb580949a3` | 多姿态起身、辅助力课程基线 |
| HiFAR | `5a5cef76eab3` | 动作空间逐级解冻参考 |
| Humanoid-Goalkeeper | `976a81ff19b7` | 后续全身门将动作先验 |
| RoboNaldo | `6ac95bd3b3af` | 后续移动球射门教师 |
| OpenTrack | `cb9b751993a2` | 本轮多姿态恢复教师与通用跟踪框架 |

讨论 4 涉及的 StableMimic、HumanUP、HoST、State-Dependent AMP、FIRM、PTDL、HiFAR、
RoboNaldo、PAiD 和 Humanoid Goalkeeper 论文均已保存到
`/code/rosclaw/rosclaw_football/papers`，共约 200 MB。

官方 OpenTrack specialist2：

- ONNX 输入：`obs[1,156]`；输出：`continuous_actions[1,29]`；
- policy SHA-256：`487ef55b522a98778a101cf957efa70ba2774262ad3adc876d7a0f1eae6c28bf`；
- config SHA-256：`3a1ac150366f78ffdf01e42c0ecfa6bda6d32a097a9c521ea5ea6909d291230b`；
- 官方动作：6 条 `fallAndGetUp`，统一预处理为 50 Hz、36 qpos、35 qvel。

原始下载、权重、视频和大体积证据均在源码仓外部，不进入 Git。

## 7. 物理场景审计与不能越过的结论边界

RoboNaldo 源场景和 OpenTrack 教师场景的 29 个 actuator 名称、顺序和关节范围完全一致；
但审计发现：

- 最大公共 body 质量差：**0.545424 kg**；
- 接触/碰撞拓扑：**不一致**；
- 因而 `scene_equivalent=false`。

所以注入试验是真 MuJoCo 物理试验，却不是原 S49 场景中的 matched promotion。

当前尚缺：

1. 不使用 reference phase / teacher id 的纯本体学生；
2. 在原 MJWarp/RoboNaldo scene 中重新训练和配对考试；
3. 新的、完整 post-save episode sealed test，而不只是已有快照附近扰动；
4. save→land→recover→`GOALKEEPER_READY` 无 reset 连续链；
5. head impact、接触冲量、关节/力矩/jerk 的完整晋级安全量测；
6. 更低峰值角速度、更短恢复时间和动作自然性。

因此本轮只允许以下表述：

> 官方多姿态起身教师能够通过失败驱动入口选择覆盖 9/9 个 S49 post-save 开发状态，
> 并在固定路由的 27/27 个局部扰动留出状态上恢复到连续稳定直立；这是可用于训练
> proprio-only recovery student 的教师桥证据，不是可部署小脑，也不是完整门将闭环晋级。

## 8. 证据与质量门

外部证据目录：

```text
/code/rosclaw/rosclaw_football/evidence/s51-opentrack-recovery-bridge-v1/
├── paired-development-exam.json
├── paired-development-exam.json.trials.jsonl
├── fixed-route-perturbation-holdout.json
├── fixed-route-perturbation-holdout.json.trials.jsonl
├── run.log
└── holdout-run.log
```

内容哈希：

- 开发报告：`sha256:a1308d7f8b9bb4880209b9199a4147eb5333a488221a590911b429fe72ae51ba`；
- 开发日志：`sha256:60e3866e4e22518e4ed0a722b6a81eb960d0c6baad8ef21546ab1628a2c53fba`；
- 留出报告：`sha256:295b57163f233594acf25b93dc80e40a208938e3f3f61b5463aedbea979e4777`；
- 留出日志：`sha256:a9c4c088ca6db32756c7ad297f90bd31782d8af3549d6f409607819223ab22cd`。

本轮已执行：

- Core continual：**79 passed**，ruff 通过，mypy 通过；
- Core 全量首次运行：**5376 passed, 64 skipped, 27 deselected, 6 failed**；其中 4 个
  LeRobot 用例由收集期用户级 runtime 与测试期隔离 `ROSCLAW_HOME` 绑定不一致导致，显式
  绑定现有 LeRobot Python 后复跑为 **4 passed**；剩余 2 个文档失败来自本轮开始前工作树中
  已存在的 `README.md` 被 MotionDecode 内容替换、`LICENSE` 被删除，本轮为保护用户改动未
  擅自恢复；
- Core 全源码 mypy：**902 个源码文件通过**；全仓 ruff 的 244 项既有问题均位于
  `examples/rh56_rps`，本轮 `src/rosclaw/continual` 与 `tests/continual` 定向 ruff 为全绿；
- Soccer S49–S51 + import contracts：**27 passed**；
- Soccer 全量：**375 passed, 95 skipped**（默认轻量环境未安装 torch 或缺少可选大资产的
  测试按声明跳过；本轮 OpenTrack 物理矩阵在其独立完整环境中实际执行，不属于 mock）；
- Soccer 全源码 ruff：通过；
- Soccer 146 个源码文件 mypy：通过。

两份 JSON 的 `report_hash` 均已独立重算一致。视频不参与这些通过判定。

### 可视化（不作为证据）

已额外复跑并渲染 3 个固定路线案例：LEFT_SIDE、唯一需要 2× 相位伸缩的困难 PRONE、
开发集中峰值角速度最高的 PRONE。三例在视频复跑中仍全部成功，单个案例内部无 reset。

- 视频：`showcase-video/s51-recovery-teacher-bridge-showcase.mp4`；
- 时长：79.72 秒；分辨率：960×720；帧率：25 fps；编码：H.264；
- 视频 SHA-256：`b7be2ac970306867f2ff4443af0d3600f9ec6e0b18a4f13d5e479dfd523ef6f3`；
- manifest：`showcase-video/manifest.json`；
- manifest 报告哈希：`sha256:96b13f0c7424d6920f3b0175301d099d0c4f4c0bc95c7edf8f4ad15e4b364532`。

画面明确标注 `SIM_ONLY`、`privileged teacher` 和 `pixels are not promotion evidence`；
它用于观察动作自然度与恢复过程，不参与 9/9 或 27/27 的数值结论。

## 9. 下一轮实施顺序

### A1：本体学生蒸馏

记录教师 rollout 中的本体历史、接触历史、teacher action、successor value；训练只看本体的
student/gate，严禁输入 reference phase 和 teacher id。先做行为克隆，再用 DAgger 覆盖学生
偏离后的状态。

### A2：原场景 4-GPU 两阶段训练

在 4×A6000 的 MJWarp 向量环境中：

1. Exploration profile 先发现 prone/side/高动量恢复路径；
2. Promotion profile 再压缩峰值角速度、恢复时间、力矩、jerk 和危险接触；
3. 训练混合采用真实快照、物理扰动、随机多姿态、dive 中间帧、Failure Memory 与
   nightmare，而不是只背 9 个样本。

### A3：sealed full-chain

用未参与训练的新 shot seeds 运行：

```text
save → flight → impact → absorb → get up → GOALKEEPER_READY → second ball
```

全程不 teleport、不 reset、不提前终止。只有 source-scene save-to-ready ≥80%、安全门通过、
历史能力不回退，Recovery Foundation 才具备晋级资格。

### B–E：再进入动态足球旗舰链

恢复基础打稳后，依照讨论 4 的 Campaign B–E 顺序，分别冻结两名队员、只训练一名：

```text
动态提前量传球
  → 奔跑中不停球高角射门
  → 全身飞扑与受控落地
  → 无 reset 三智能体完整链
```

这保证每次“成长”确实来自焦点智能体，而不是射手变弱让门将数据看起来变好。

## 10. 通俗解释

以前像是教机器人在床垫上仰卧起床，单项考试满分；真正门将扑完球却趴着或侧躺在草地
上，于是它完全不会。现在我们先请来一个见过多种倒地姿势的“体操教练”，让系统根据
门将真实倒地状态找最合适的教学动作。只找外观最像的动作能救回 7 个；让失败结果反过来
选择更合适的动作能救回 8 个；最后一个把教学动作放慢后也能站稳。

随后我们把答案锁死，再轻微改变每个倒地状态考 27 次，全部通过。这说明教练确实提供了
可迁移的起身知识。但比赛时机器人不能一边起身一边偷看“教学视频播放到第几帧”，也不能
换一块草地就失效。下一步就是把教练示范压进只靠自身感觉工作的“小脑学生”，再回到原
足球物理场训练。那时才从“教练牵着能做”变成“自己真正学会”。
