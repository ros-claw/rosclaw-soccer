# 完整脚掌任务接触：根因检查与新物理协议

## 结论边界

还没有足球能力突破，也没有新 PPO 学习成果。当前推进的是更基础的问题：
物理模型是否允许可见脚掌正常接触球。不能因训练艰难就更换成功指标，
也不能把修正碰撞体说成学会了踢球。

## 为什么停止旧世界 PPO

实际编译世界 `246da475bbad8d3ced351f6ceb18e18fc3d4c2de7236cdd8c427239cecb4ff83`
里，两脚可见 mesh 都是 contype=0/conaffinity=0，每脚仅四个半径 5mm
支撑球与足球接触，npair=0。球仍为半径 0.11m、质量 0.43kg。
左脚可见 mesh 原尺寸约 20.8×7.6×5.9cm，不能放大成“隐形球拍”。
检查只读取 MJB，零积分步；证据 `rsi-foot-ball-contact-inspection-20261018.json`，
seal `f6d9a6c8a3e93f0cf20c412d97e370c52d57257820613407eeafd278d49f9e10`。

这不证明所有失败都由接触面造成。32 条新状态动作选择实验中，三种 seed
都是 3/8 通过，对照 1/8；新增通过仅来自一个父场景簇；还存在胫部碰球、
不触球和过晚触球。不能称为广泛泛化。`knee_link` 含小腿网格，不能仅凭
body 名称把接触点叫成膝盖。原 trace 中所谓 foot_geometry_position 实际
是四个 body 原点，也不是脚面最近点。

旧 44-body PPO 阶段终止于 2 个扣费 attempt、1 个合格 zero control、
另一个中断，更新 0 次。V2 zero control 已通过原有公共数组字节中性验证。
中断属于原实验失败终态，不能重置、删除或原地续写来掩盖。
另一动作选择实验仍按冻结声明完成 retention，不热改它。

## 可复用实现，而非脚本造球

`physics.task_contact_surface.with_mesh_task_contacts` 接受任务物体几何名
和表面 body 名，复制 MjSpec 后复用现有可见 mesh，不新增几何、不缩放。
增加 mesh↔球显式 pair，同时排除该脚 body↔球自动接触，避免同时使用
支撑点和完整脚面。原地面支撑、惯性、关节、执行器、几何位姿和碰撞 masks
保持不变；构造时检查 28 个编译数组。歧义表面、已有 pair、膨胀 margin、
不一致 solver 参数均拒绝。旧八支撑点接触特征在新 mesh 世界 fail closed。

MuJoCo 显式 pair 绕过自动 body exclusion/mask 过滤，因此能做到只替换
特定任务物体接触；但 mesh 碰撞使用凸包，不是精确凹网格。
见 [官方碰撞说明](https://mujoco.readthedocs.io/en/stable/computation/#collision)。
显式 pair 需要独立指定接触参数，不能假设自动继承；本实现保留相容的
solver 设置，并采用原材料摩擦系数最大值的五维展开。
见 [官方 pair 文档](https://mujoco.readthedocs.io/en/latest/XMLreference.html#contact-pair)。

## 已完成的验证

62 项相关测试通过（新构造器、旧特征、接触力和动力学）；不是全仓测试。
微型物理夹具确认完整 mesh 可触球、原支撑点不再重复触球、球仍与地面碰撞。
无球干扰时 1000 步地面动力学五类测试字段字节相等。

实际 G1 已进行一项 6 秒、每世界 3000 子步的原始力矩回放诊断：
重建旧 MJB 哈希与原模型完全一致，新世界哈希
`df743dfc121a2649bc9159fa38cf2b6a1c7d618113484b882bce25cbd3e1601e`。
球移到远处，两世界 qpos/qvel/qacc/qacc_warmstart/qfrc_constraint/
actuator_force 每步字节相等，总计 6000 新积分步。
证据目录 `rsi-mesh-contact-ground-parity-20261018-preflight`，结果 seal
`13d2efc9b016590820b6dfe0d15d19791e88b7013fdcc4e96eefad68ee3a7696`。

**重要反例**：原始力矩开环回放离开原来的球反馈后，两世界都倒地，最小
root 高度约 0.05175m。这只说明修改没有改变这个配对动力学轨迹，不证明
无球平衡、步态质量或新接触世界踢球成功。不得隐藏倒地来称“稳定性通过”。
此预检运行固定源码哈希并结束重验，但并非 immutable git checkout；正式
后续诊断使用冻结提交。构造 receipt 的 physics_qualified 仍为 false。

已在冻结提交 `d9b09fd` 上完成 300 个记录状态各 20ms 的 old/new 分支：
总计再增加 6000 子步。旧世界接触出现在 frame 85–88，新世界 84–88；
旧/新右脚峰值力分别 196.79/199.71N，所有记录的非地面球接触都属于右脚
ankle_roll body。两分支末端球速度最大差异 0.47022m/s。这证明接触变更
确实影响局部物理，**不证明完整 episode 或安全性更好**。分支每帧重置
warmstart，未来目标和力矩未重新规划，不能把各帧拼成自主成功轨迹。
结果 `rsi-mesh-contact-branch-diagnostic-20261018.json`，seal
`af2c07370f0b60c2340ea75159f32ac5d9e0947ab56ff069f6ab1bc8576370a8`。

新增 native 显式新世界入口：必须给出预先编译的完整 MJB hash、捕获真实
Foundation 输入且处于 development diagnostic；在 Foundation 分配前验证
新世界 hash。旧参数 None 时不改碰撞世界或控制算式。此初始入口拒绝混入
旧小脑/采样或宣称 public training，先完成接触世界独立资格再开放学习。
qualified_episode 仍对默认路径强制旧世界；新路径需独立参数和 receipt，
不改 assets.WORLD，也不将旧资格自动转移。计划固定三例：group64 option3、
group96 option3、group104 option7；不是广泛性能考试。

回归检查发现原冻结 AST 测试期望值已落后于此前提交的小脑可选层。
本次将新 mesh 选项严格专门化为 None 后，与父提交 d9b09fd 的规范化 AST
逐字节一致（hash `65553cdb7c835e95fd63a2a20d987ebdb9237aaae1d7c17e4e4a07915cab88ce`），
据此更新基线；不是拿新控制算式的任意 hash 放过回归。245 项相关测试通过，
包含新增早期拒绝测试；不冒充全仓通过。

## 后续阶段和拒绝条件

1. 冻结新源，在原始 300 个状态各做 old/new 20ms 配对分支，记录真实几何
   ID、接触点、带方向力与球速度。这是局部物理诊断，不是 300 个独立球场。
2. 无球干扰下增加相同关节目标、实时身体 PD 的配对测试；再接入真实当前
   身体 Foundation 闭环，而不是照抄未来力矩。
3. 新世界重新声明有限数据和课程，验证独立 CPU 重演及接触资格，之后才
   采集新 on-policy actor-critic 数据。旧 world hash 约束不放松。
4. 多 seed、保留集、失败回收、连续身体恢复都通过后，才扩展到球队角色。

持续实施依靠 active Goal 自动接续与有限 systemd 工作者，不靠假心跳。
观察器只报真实 PID、CPU 和账本，不生成学习，不保证断电时仍能运行。
所有实验 SIM_ONLY，不授权硬件、自动策略晋升或宣传成果。
