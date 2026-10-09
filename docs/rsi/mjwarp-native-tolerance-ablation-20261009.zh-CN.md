# MuJoCo/MJWarp 容差与反馈对照实验（2026-10-09）

## 结论与影响

恢复原始求解容差能通过转换参数一致性检查，但**没有解决完整
轨迹分叉**。这不是球队能力提升、GPU 足球训练资格或宣传成果。
现有 CPU MuJoCo 完整考试、160-episode 课程采样及后续学习队列
没有改输入、没有取消，也没有混入这些 GPU diagnostic 数据。

本轮的实际进展是完成一个可复现、配置受控的后端反事实实验，
排除“只恢复容差就足够”的简单方案。它不能判定所有误差的唯一
原因，更不能证明真正的闭环神经策略在 GPU 上必然失败。

## 为什么需要做这个对照

当前物理真值是 CPU MuJoCo strict replay。GPU 并行仿真能帮助
扩大训练量，但必须先验证物理与任务标签，不能只凭导入成功、
短时小误差或 GPU 利用率来接入学习银行。

前一轮发现官方 MJWarp 3.13 转换会把 native tolerance 的下限
静默设为 1e-6，而原 G1 MJB 是 1e-8。ROSClaw 已增加转换选项
核对，原官方构建会在 forward、MjData、policy 与 physics step
之前被拒绝。此处不关闭该 guard，不放宽 CPU 世界参数。

## 唯一受控改动

参考来源是官方 [MJWarp v3.13.0](https://github.com/google-deepmind/mujoco_warp/tree/v3.13.0)，
原提交 `d1a55b6deb7dc541aad2f10067cb287db0357a2b`。
本地隔离提交 `c077b820355a0627125a17655d32e780e729b608`：

- 转换时保留 `mjm.opt.tolerance`，不再静默取 1e-6 下限。
- 版本改为 `3.13.0+rosclaw.native.tolerance`，避免冒充官方发行版。
- 锁文件仅同步该本地版本号，未改变依赖版本。

[诊断补丁](patches/mjwarp-v3.13.0-native-tolerance-diagnostic.patch)
只供隔离研究，不是默认安装配置或已经获准的训练后端。其源自
Apache-2.0 上游；使用时须保留原项目许可及署名，参见上游
[LICENSE](https://github.com/google-deepmind/mujoco_warp/blob/v3.13.0/LICENSE)。
补丁在原始官方提交的独立临时 index 中 apply-check 通过，
在修改版上 reverse-apply check 也通过，未改变真实 index 或
working tree。没有向上游仓库推送或创建 PR。

环境固定为 MuJoCo 3.12.0、Warp 1.18.0、NumPy 2.3.5，其余
依赖也与前一轮 979 环境一致。独立安装后的 io.py 与固定源码
逐字节相同，SHA256 为
`97f29a1f527f01c74ea3a45c86ad258809805d902a1ecf9c96d7e71ae3598db6`。
CPU 动态库仍为
`bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8`。

## 课程与控制边界

全部对照使用原 960 case0 的同一个 MJB、同一个初态、同一组
300 帧 joint targets。没有移动球、改球尺寸、改摩擦、改碰撞体、
改原始 PD gains、改 torque limits 或改求解器迭代数。

每个实验是 8 个 GPU 世界副本 × 3000 个 500Hz substeps，
并有 3000 步原 CPU scratch replay。这是**一门课程的副本**，
不是 8 门独立课程；三个实验亦不算 24 门独立课程。

981 直接重放归档 torque。983/985 按各 GPU 世界自己的当前
关节状态计算 PD，但目标仍来自归档，因此是 teacher-forced
target replay，不是自主 Foundation/GRU/actor 的神经策略闭环。
两者完整 PD step-loop 的 AST 一致。985 每个 CPU substep
必须精确重建原 torque，原初态及随后 299 个状态也必须 exact。
最后 after300 状态没有原归档对应项，未冒充第 301 帧真值。

所有结果固定 `teacher_forced_replay_eligible_for_learning=false`、
`full_episode_backend_qualified=false`，不产生 RL updates、
新 actor、私有 Fresh 或晋升。视频也不能代替这些资格。

## 完整回放结果

下表时间为控制帧结束后的状态，第 f 帧时间为 f×0.02s。
阈值仅用于定位分叉，不是事后补设的合格门槛。pelvis 最低值
来自保存的 50Hz 帧，不冒充全部 500Hz safety samples。

| 对照 | 控制方式 | root 误差首次 >0.1m | 球误差首次 >0.1m | 最大球位置分量误差 | sampled pelvis 最低值 |
| --- | --- | --- | --- | --- | --- |
| 981 | 归档 torque 开环 | 60 帧 | 65 帧 | 15.8993m | 0.06086m |
| 983 | PD 反馈，官方 3.13 | 127 帧 | 104 帧 | 8.8077m | 0.07429m |
| 985 | 同样 PD，保留 native tolerance | 111 帧 | 97 帧 | 8.8208m | 0.05974m |

CPU sampled pelvis 最低为 0.69660m，最终球 x 约 9.603m。
985 的 8 个副本末球 x 分别约为
`[0.782, 0.782, 4.219, 0.801, 9.371, 0.800, 0.815, 10.612]m`。
不能挑第 4/7 副本当宣传成功，也不能从单一课程断言新构建
整体更优或更差。副本间也非 bit exact，需要在正式 label
agreement 中认真处理，不能隐藏这种差异。

985 的实际转换参数：dt≈0.002、tolerance≈1e-8、line-search
tolerance≈0.01，Euler、Newton、100 solver iterations、50
line-search iterations、cone/enable/disable flags 均与原 CPU
选项相同，浮点仅有精确 float32 表示差异。10 项参数检查通过，
却仍出现明显物理分叉，说明配置门是必要条件而非充分条件。

## 证据与复查

外部证据 ID：`rsi-mjwarp-native-screen-probe-v985`。
Runner SHA256：
`46a5af5bf9036ee6865bb66662df94e36d46c31a97d70db9588381edc676aaef`。
Result seal：
`d1e2641d6eec640e4b62d7702c345bcdf9a2a4c5a5f39e9cdffdceb10c37549b`。
原 MJB SHA256：
`246da475bbad8d3ced351f6ceb18e18fc3d4c2de7236cdd8c427239cecb4ff83`。

正式执行 exit0，actual GPU world steps24000、native CPU
steps3000。exit0表示诊断执行完整，不表示后端合格。
独立只读复核 exit0，核对完整 seal、全部 2223 项 input bytes、
三个 primitive NPZ hashes、全部状态 max error、10 项转换
参数、2400 份 frame-start PD commands 和 24000 份有界
GPU torque vectors，均一致。所有失败/分叉数据均保留。

预检说明：首次安装受上游 tool.uv.sources 的 nightly 源选择
阻止，未安装成功；改用 `--no-sources` 与明确版本后成功。
参考项目要求的 pre-commit install 已执行；首次 commit 因
uv-lock 同步本地版本而 exit1，复查仅一项版本变化后重新提交，
ruff、format、uv-lock、kernel-analyzer 全通过，没有跳过 hook。
不声称执行了其全量 pytest 或创建上游 PR。

985 初次 source-only 预检结束后，复查发现缺少 upstream HEAD
显式断言；在正式执行之前补上两个干净 HEAD 的检查，重新
lint/预检通过后才启动。没有热改运行或等待中的源码/环境。
所有源码固定后才执行 physics，完成时再次检查输入字节与 HEAD。

下游相关七组测试复查 147 passed、6 skipped、2.40s，修改模块
mypy 与相关文件 ruff 通过；这不是上游或下游的全量测试通过。
6 个 skip 仍为 5 项 opt-in GPU 测试与 1 项缺 G1 assets。

## 下一步决策

不把这个配置当 GPU 足球训练后端，不修改现有 CPU 学习链。
提交前原 963 考试完成 29/52 例，exact owner 仍运行，没有全量
终态。965/971/972/973/974/975/976 exact owners 也均存活，
等待完整上游结果，不因为观察超时重启、删失败或选成功子集。
继续等待完整 52 例考试及后续全部 160 例采样、成对学习、
独立复核和完整模型导出。新候选仍须重新证明物理增益与保留性。

若继续 GPU 路线，下一项必须是使用各世界自身观测与 recurrent
state 的完整 Foundation/actor 闭环，以及事前声明的分离、平衡
课程 label agreement；不能继续把归档 targets 或 torques 当
自主 rollout。即使该阶段通过，最终物理真值仍由 CPU strict
replay 给出；M0、球队自主比赛和可宣传连续视频目前均未完成。
