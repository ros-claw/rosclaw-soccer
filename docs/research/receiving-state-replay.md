# M0 完整初态：物理恢复与控制器恢复分开验收

本记录是 Receiving Mechanism Reboot 的工程接续，不是接球突破报告。

## 为什么不能直接复用旧轨迹作为初态

旧课程保存的 qpos/qvel 与观测轨迹适合评分，但不能据此声称能恢复整个比赛循环。
当前 `skills/team/independent_team_world.py` 还持有以下状态：

- MuJoCo 时间、激活、控制输入、求解器热启动、外力、mocap、userdata 等 integration state。
- 每名球员的 locomotion policy 记忆、上一动作、坐标反射状态、动作选项生命周期、踢球与接球历史。
- `residual_rng` 和 `residual_previous`。
- 接球租约、持球与争球分配、触球时间、handoff、capture hold、射门租约。
- 球网状态、裁判状态、决策时钟、轨迹评分累计量。

同一机器人位置不代表同一可继续执行的系统状态。A0/A1/A2/A3 必须使用相同物理状态，但各基础策略需要与其输入契约一致、来源明确的历史初始化；不能把 A0 的神经隐藏状态直接作为 SONIC 的状态。

## 本次实现的最小原语

`sim.physical_checkpoint.PhysicalCheckpoint` 保存 `mjSTATE_INTEGRATION`，绑定实际编译模型的 MJB 哈希及 MuJoCo 版本，使用不可变数值字节，拒绝损坏、非有限值和物理参数变化。

恢复创建独立 `MjData`，不修改正在运行的世界。它不自动执行 `mj_forward`：刷新派生字段可能改变 warm-start，调用方需要在完整闭环回放中明确验证刷新过程。刚恢复后的 contacts/sensors 不能当成已经刷新的观测。

**这不是完整控制器快照 API。** 目前不能使用该原语把 64 个物理副本计为已完成的 64 个 M0 案例，也不能据此启动教师训练。

## 验证设计

1. 球—地面接触模型：恢复后连续 200 步，每步完整 integration state 完全相同。
2. 有激活动力学的执行器：control、activation、warm-start、外力、mocap、userdata 恢复，后续 20 步完全相同。
3. 相同维度但改变摩擦、版本不符、数据损坏、长度错误、NaN：拒绝。
4. 外部 `probe_physical_replay.py` 在既有 G1 接球课程中插入 64 次独立单物理步分支；保持原课程和完整重跑的逐数组一致性检查。结果以外部 `m0/physical-fork-probe/verification.json` 为准，未产生该文件不能声称通过。

第 4 项只检验同一课程不同时间的物理分支，不是 64 个独立任务，不检验 controller 的跨进程恢复，不改变 R0，不产生学习更新。

## 本轮实际结果

外部 G1 probe 已退出 0，64 次分支的下一步完整物理状态全部精确相同，原课程和整段重跑的所有轨迹数组也完全相同。
这是恢复基础设施验证，不是新增 64 次接球成功。该 probe 源码版本已保存在外部 `implementation-archive/physical_checkpoint_probe_v1.py`。
随后补充了 capture 时拒绝不属于传入模型的 `MjData` 的前置校验；该校验由专门单元测试覆盖，不将旧 probe 的源码哈希冒充最终版本。

## 下一项必须通过的门

将 M0 接球微环境的可变状态显式封装，保存策略绑定、随机数、上一动作与参考历史。在控制帧边界暂停后，独立恢复并完成至少一个完整接球评分窗口；要求请求动作、执行动作、物理响应及最终评分全部一致。
先用一例证明，再扩展至站立／运动、左右支撑及边界初态；不要通过序列化整个 Python 进程或 pickle 外部对象掩盖遗漏。
