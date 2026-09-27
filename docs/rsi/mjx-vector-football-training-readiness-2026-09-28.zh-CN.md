# 四卡 MJX 足球物理训练入口：可复核准备度

状态：`SIM_ONLY / TRAINING_READINESS_ONLY`。这一步解决上轮“不能只在常数速度上扫参数”的基础设施缺口；**尚未训练新的残差 actor，更未通过传接射扑或球队宣传门**。

## 物理模型与差异边界

原 RoboNaldo G1 + 标准尺寸仿真球场模型包含 5 个只读框架/IMU 传感器，当前 MJX 在其中一个传感器类型上运行失败（`KeyError: np.int32(6)`）。新增 `build_g1_stadium_sensorless_model` 仅在编译前去掉这 5 个传感器，保留相同 G1、球、地面、球门、执行器和碰撞几何；它只能用于训练计算，最终物理事实仍需原始 CPU MuJoCo 模型验收。

实际资产 `/code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy` 上，测试比较两模型的质量、惯量、几何、摩擦、自由度阻尼和执行器增益/偏置逐值相等；同一初态与零扭矩控制下连续 100 个 CPU MuJoCo 物理步的全部 `qpos/qvel` 逐步一致（`atol=1e-12`）。这证明**剥离传感器本身**不改变该课程的 CPU 动力学，不证明 MJX 与 CPU 在任意接触过程中永远等价。

先在 A6000-0 上跑 4 个不同初始球 x 坐标的独立 MJX 环境，每个真正前进 10×2 ms 物理步；与 CPU MuJoCo 同初态首环境比，最大 `qpos` 差 `1.92e-7`、`qvel` 差 `3.26e-5`。之后分别绑定物理 GPU 0、1、2、3，每卡 16 个环境、每环境 10 步，四进程全部退出码 0；源脚本 `scripts/rsi_mjx_sensorless_smoke.py` 把可见物理 GPU 编号、场景/源码哈希、物理状态和 CPU 对照写到外部 JSON：

- `/data/rosclaw_overflow/rsi-mjx-sensorless-smoke-gpu0-20260928-v2.json`
- `/data/rosclaw_overflow/rsi-mjx-sensorless-smoke-gpu1-20260928-v2.json`
- `/data/rosclaw_overflow/rsi-mjx-sensorless-smoke-gpu2-20260928-v2.json`
- `/data/rosclaw_overflow/rsi-mjx-sensorless-smoke-gpu3-20260928-v2.json`

四份报告分别记录 `physical_gpu_index=0..3`、`envs=16`、`gpu_physics_advanced=true`；各首环境 CPU/MJX 位置最大差约 `1.93e-7`、速度最大差约 `3.27e-5`。**这些是四组独立批量物理烟测，不是四卡梯度同步 PPO，也不是 64 条训练轨迹。** JAX 安装了与本环境 0.11.1 匹配的 CUDA 13 PJRT/plugin；运行显式设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，避免默认抢占其它 GPU 作业的显存。

## 下一训练阶段的硬门

已从同一 RoboNaldo MuJoCo 场景采集冻结 SONIC 1.4 m/s 的 300 帧目标轨迹 `/data/rosclaw_overflow/rsi-mjx-parent-runup-20260928-v1/`。其球虽越线，但右小腿先碰球（约 2.024 s）、左脚晚到（约 2.238 s），**不能当成功教师**，仅能作为冻结全身行走基底。下一步用该底座在 MJX 并行环境上训练受限脚/膝残差 actor：观测含短时本体和球/足/膝相对状态，动作有明确关节目标权限，奖励必须由物理脚触球、出球方向与安全联合驱动，并显式惩罚小腿先碰和触后膝二次碰；训练集与新鲜保留集按场景组拆分。必须先证实 50–300 帧滚动球接触和 CPU 标签一致，才能扩大批次或晋升。当前“一控制帧无接触烟测”绝不支撑任何射门成功率主张。

## 追加：真正触球的 CPU↔GPU 诊断

新增 `scripts/rsi_mjx_contact_replay_smoke.py`，从上述冻结 SONIC 轨迹第 80 帧的本体状态出发，连续回放 30×10 个 2 ms 物理步，并以相同 PD/力矩上限在 CPU MuJoCo 逐球位对照。脚、非脚是按球与 G1 碰撞几何判断，不以球位移冒充触球。首次四球位报告中 MJX 曾漏掉 CPU 的一帧脚触球；后来同一球位重复运行可出现这帧，且 GPU 整条 `qpos` 哈希随重放改变。因此“导入 MJX 成功”和“整条 GPU 轨迹严格确定”都不能声称。

已消费的 16 球位诊断网格为 `x={2.08,2.16,2.24,2.32} m` × `y={0.06,0.14} m` × 初速 `v_x={-0.5,+0.5} m/s`，滚动自旋与速度匹配；GPU 0 上 16 环境并行 30 帧退出码 0。CPU 类别分布：纯脚 4、仅非脚 4、脚与非脚都有 1、无机器人触球 7。第二次网格报告中 MJX 对**整回合是否出现脚/非脚接触**为 **16/16** 一致，但严格逐帧事件门失败：一个球位非脚接触 4 帧中仅 3 帧时间重合，且接触时间可差约 0.12 s。该报告在 `/data/rosclaw_overflow/rsi-mjx-contact-grid-20260928-v2.json`，哈希 `sha256:25774cad698aaeca391d477893335ebed21cea07ebaaeeb3ed7242c276a22994`。

尝试 `JAX_ENABLE_X64=true` 没有获得更严谨的数值比较：当前 MJX 在扫描物理步时内部接触索引由 int32 变 int64，JAX carry 类型不一致而拒绝编译；这是**失败**，不是双精度通过。后续不通过关闭检查或删接触来绕过。MJX 可作为并行候选筛选和以整回合类别为主的带噪训练信号，但脚面力、细时间窗和最终成功必须交给 disjoint CPU MuJoCo 严格物理重放；本 16 球位已消费，不能再充当新鲜或 Sealed 集。尚未有新 actor 的训练成绩。
