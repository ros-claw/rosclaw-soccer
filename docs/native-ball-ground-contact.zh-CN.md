# 仿真足球与地面接触审查

`inspect_ball_ground_contact` 读取 MuJoCo 球—地面接触项的维度和有效滚动摩擦系数。调用方须提供匹配的模型及在当前状态完成 `mj_forward` 的数据；`mj_step` 后保留的接触项可能对应积分前阶段，本函数无法判断缓存何时刷新。它不修改状态、不产生控制命令。

没有观测到球地接触时返回 `NO_OBSERVED_CONTACT`，不能当作没有摩擦；仅有因 gap 而未进入约束求解的近接触时返回 `INACTIVE_CONTACT_ONLY`。实际三维接触返回 `NO_ROLLING_CONSTRAINT`；六维且存在非零滚动参数返回 `ROLLING_CONSTRAINT_PRESENT`。应读取实际接触项，而不是只检查两个几何体的默认参数，因为显式 contact pair 可以覆盖几何体配置。具体语义参见 [MuJoCo 接触维度文档](https://mujoco.readthedocs.io/en/latest/computation/)。

此审查只说明当前模拟器是否具备指定接触约束；不证明球正在滚动、一定在减速、摩擦参数与真实球场一致、场景物理准确、球员任务完成或真实机器人安全。滚动姿态可另用已有 `measure_rolling_authenticity` 检查；球速衰减需要真实时间轨迹和单独的对照或标定，不由本函数推测。默认场景和冻结实验均不自动更改。
