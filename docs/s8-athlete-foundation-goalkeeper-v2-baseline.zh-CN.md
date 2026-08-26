# S8 Athlete 基础与 Goalkeeper V2 父策略基线实施报告

日期：2026-08-13
证据域：`SIM_ONLY / CPU_MUJOCO`
状态：完成 S8.1–S8.6 的首轮工程闭环；未训练、未晋升 Goalkeeper V2

## 1. 本轮结论

本轮的重点不是继续微调某一次射门，而是把 ROSClaw Soccer 从“三个控制器”推进到
“三个能够独立成长的球员”。已经完成：

1. ROSClaw Core 通用数值可复现合同；
2. ROSClaw Core 通用 `IndividualGrowthScope`；
3. Soccer 三名球员各自独立的 lineage、memory、Parent/Candidate/Champion；
4. 从目标动作到真实关节加速度的 `Agility Profiler`；
5. Goalkeeper V2 因果 Actor 观测合同，移除射手内部 `policy_frame`；
6. 5 档时限 × 5 个区域的 Coverage–Time CPU MuJoCo 基准；
7. 25 个场景双跑、逐 trial 严格重放一致的外部 evidence。
8. train-only Conditional Teacher Prior V2：强制以 `task + region` 选择动作教师。

当前门将 V2 只是**合格的父策略基线**，不是训练成功的候选：慢球中心区域能挡，四角球
和 0.6 秒以下高速球基本无能为力。这一失败已经被量化，下一轮 PPO/条件动作先验有了
明确且不可通过“射手变弱”作弊的优化目标。

## 2. ROSClaw Core 新增的通用能力

### 2.1 NumericalRuntimeContract

新增 `rosclaw.continual.reproducibility`：

- 固定 OMP、OpenBLAS、MKL、VecLib、NumExpr、BLIS 线程数；
- 固定 `PYTHONHASHSEED` 与 cuBLAS workspace；
- 固定 ONNX provider、串行/并行模式、intra/inter-op 线程；
- 固定 deterministic compute、TF32 和浮点模式声明；
- 环境缺项与值不一致都 fail closed；
- 只生成子进程环境，不在数值后端导入后偷偷修改当前进程。

本轮证据使用单线程 CPU 合同，合同哈希：

`sha256:85051a841932699b3134ef92dff1b0febf9890453ebffcec48f3248db268477a`

### 2.2 IndividualGrowthScope

新增 `rosclaw.continual.individual_scope`，与足球无关，可供机械臂、移动机器人、多台人形
机器人等复用。一个 scope 显式绑定：

- agent、body、body state；
- shared foundation；
- personal adapter、role policy、residual policy；
- personal memory、failure memory；
- capability profile、career lineage；
- Parent、Candidate、Champion；
- frozen partner snapshot；
- matched-seed promotion evidence。

Candidate 必须以本 agent 的 Champion 为 parent，且 body、controller、safety kernel 不能
在一次个人候选中漂移。晋升证据若属于另一个 agent、另一个 memory namespace、另一批冻结
同伴或另一个 Candidate，都会被拒绝。

## 3. Soccer 独立球员架构

`SoccerPlayerProfile` 将三个角色正式分成：

- `soccer.finisher`：终结者；
- `soccer.playmaker`：组织者；
- `soccer.goalkeeper`：门将。

三人共享一个 Athlete Foundation，但 Champion、personal adapter、role/residual policy、
个人记忆、失败记忆和职业 lineage 全部分离。`SoccerTeamRoster` 另行版本化战术策略和阵容；
个人晋升只替换一个球员，不能顺带更新另两人或公共基础。

这使“球员升级”和“球队升级”成为两件不同的事。

## 4. Agility Profiler

Profiler 每个控制周期分析：

```text
policy desired motion
  -> target q / dq
  -> PD commanded torque
  -> safety projected torque
  -> actuator executed torque
  -> actual joint acceleration
```

同时输出：

- reaction latency；
- skill handoff latency；
- motion pause time 与 idle ratio；
- target velocity clipping；
- safety torque projection fraction；
- actuator tracking miss fraction；
- 目标/实际关节速度、实际关节加速度；
- inference latency p50/p90（没有 wall-clock telemetry 时明确缺失，不伪造为零）；
- 策略、Safety Projector、执行器/动力学与动作停顿四类瓶颈归因。

三人同场轨迹现已补齐门将 commanded/projected/executed torque、target velocity、观测到的
飞行时刻等通道。

## 5. Goalkeeper V2：消除隐藏相位泄漏

历史门将通过射手内部 `policy_frame` 预判和触发 block action。这对调试有效，但不符合独立
Actor 的部署边界。

V2 Actor 只接收：

- 8 帧稳定门将锚点坐标系下的足球位置历史；
- 重力方向、根部线速度、角速度；
- 自身 29 个关节位置与速度；
- 上一动作。

Actor 不接收球速真值、拦截点真值、目标区域、接触真值、射手状态或任何其他 policy phase。
训练 Critic 可额外使用这些全局真值，Actor/Critic 合同拥有不同哈希。Actor 合同哈希：

`sha256:69fb951855dafe9d81d11a00be4fe7c9956453d6932d9cbcad5eb8d1df0654d5`

开发中连续发现并修复了两个因果观测错误：

1. 用固定局部 x 判断来球，会把前锋传球误判为射门；
2. 用当前身体相对坐标历史，会把门将自身待机晃动误判成足球在接近。

最终使用稳定的“门将初始锚点/球门局部坐标系”记录球历史，既不读取世界未来真值，也不被
自身待机摆动污染。真实三人 rollout 中，射门接触发生在 7.594 s，V2 从位置历史确认飞行
发生在 7.640 s，随后按父策略的 80 ms 反应延迟触发。

## 6. Coverage–Time 基线

基准使用一个冻结、确定性的 MuJoCo ball launcher，其他条件完全相同，只改变来球时限和
目标区域。目标区域为左上、右上、左下、右下、中心；时限为 1.0、0.8、0.6、0.5、0.4 s。

| 来球时限 | Coverage | 接触率 | 真扑救率 | Reaction p50/p90 | 横向预测误差 | 最大安全成本 |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 s | 20% | 20% | 20% | 80/80 ms | 6.17 cm | 0 |
| 0.8 s | 20% | 20% | 20% | 80/80 ms | 6.70 cm | 0 |
| 0.6 s | 0% | 0% | 0% | 80/80 ms | 7.41 cm | 0 |
| 0.5 s | 0% | 0% | 0% | 80/80 ms | 8.54 cm | 0 |
| 0.4 s | 0% | 0% | 0% | 80/80 ms | 36.16 cm | 0 |

父策略只能挡住 1.0/0.8 s 的中心来球，对四角均失败。0.6 s 以下完全没有覆盖能力。
这不是宣传成绩，而是 S9 Goalkeeper PPO + conditional motion prior 必须显著左移、上移的
冻结父曲线。

尚未测量并明确保持 `null/0` 的能力：Human Motion Score、扑后恢复时间、二次扑救。任何
候选在这些字段补齐前都不能宣称“像人的全身门将”或完成晋升。

## 7. 严格证据

最终 evidence：

`/code/rosclaw/rosclaw_football/evidence/s8-goalkeeper-v2-parent-baseline-v3/`

- `request.json`：运行合同、参考边界、trial 数量；
- `goalkeeper-v2-parent-baseline.json`：25 条 trial、聚合曲线和声明边界；
- 25 条场景完整运行两次，逐 trial 字典一致；
- `strict_replay = true`；
- evidence hash：
  `sha256:89ff3e427a6186847d005b7ce78765d79e9158cdef04e24acab4af7e437b9980`；
- report hash：
  `sha256:2426610d06d2558d75880c57710c796332e460c4a98dc025790c8631f21fa10d`；
- Actor observation contract hash：
  `sha256:69fb951855dafe9d81d11a00be4fe7c9956453d6932d9cbcad5eb8d1df0654d5`；
- `promotion_status = BASELINE_ONLY_NOT_CANDIDATE`；
- 没有像素评分、硬件命令或 REAL 权限。

## 8. 参考项目与许可证边界

研究参考固定为官方 Humanoid Goalkeeper：

- 仓库：`https://github.com/InternRobotics/Humanoid-Goalkeeper`；
- commit：`976a81ff19b7306bafbe923d2890066b68a85271`；
- 许可证：CC BY-NC-SA 4.0。

由于该许可证与本仓库 MIT 发布边界不同，本轮只 clean-room 借鉴“位置条件动作先验、非对称
Actor/Critic、延长恢复 Episode、Coverage–Time”思想，没有复制其代码、权重或数据。

## 9. 本轮顺带修复的集成问题

同步最新 ROSClaw Core 后，soccer 的 recovery 路径假定每帧 effect 一定包含 terminal PD
damping 字段；当前 Core 版本并不提供，导致真实 MuJoCo rollout 崩溃。现已在共享三人世界和
单人 free-kick 两条路径加入身份兼容语义：旧 Core 缺字段时使用 `kp=1.0/kd=1.0`，不改变
冻结轨迹；新版字段存在时照常使用。

## 10. 下一轮实施入口

Core 已新增 Conditional Teacher Prior 通用合同；soccer 已冻结 `ready/shuffle/save/landing/
recovery × center/upper-left/upper-right/lower-left/lower-right` 词表，查询若缺任一 condition 或
试图使用“average”教师会 fail closed；部署 Actor 依赖教师也会被拒绝。

下一轮不再调旧门将的横移增益，而按以下顺序推进：

1. Goalkeeper Motion Library：ready、shuffle、左右高/低扑、落地、起身；
2. 4 卡并行 PPO + position-conditioned AMP，Actor 使用本报告的因果合同；
3. continuous reset：从扑救中间态、失败态、落地态启动；
4. 每个 Candidate 与冻结 launcher/射手、相同 seeds 做差分评估；
5. CPU MuJoCo 重跑本报告 5×5 曲线，先达到相对父策略的硬门：coverage、contact、save 上升，
   reaction、recovery 下降，安全零回归；
6. 再加入二次扑救与 30 秒不 reset 连续视频。

这轮完成的是“计量、身份和因果边界”。它不会直接让旧门将变成球星，但已经消除了继续做
伪成长的三个主要入口：数值环境漂移、球员间更新污染、Actor 偷看射手内部状态。

## 11. 验证结果

- ROSClaw Core `tests/continual`：53 passed；
- ROSClaw Soccer：178 passed、4 skipped（均为已注明的堆叠 PR/历史迁移条件）；
- Core 新模块 `ruff`、`mypy --strict`：通过；
- Soccer 全源码与测试 `ruff`、77 个源码模块 `mypy`：通过；
- Coverage–Time：25 个场景完整运行两次，共 50 个 CPU MuJoCo 回合，逐 trial 严格一致；
- evidence 文件声明哈希已从落盘 JSON 独立重算并一致。
