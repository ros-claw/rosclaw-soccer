# 共享球场足端接触：机制成功，泛化失败（2026-09-29）

## 判定

这是 **SIM_ONLY** 的 3v3 MuJoCo 物理实验，不是宣传片，也不是可晋升的球员策略。已证明一件很窄但真实的事：在一个预先消费的来球场景中，绑定 G1 当前足端位置、雅可比与关节限位的任务空间动作，让红队组织者以左脚碰到球；相同共享球场 Parent 没有碰球。触球后球的 x 速度约从 `-0.321` 变为 `+0.145 m/s`，只是一次较弱的反向拨球，**不是有效传球或射门**。随后冻结此动作，在六种邻近来球位置/速度上成对测试，安全脚球接触 **0/6**、达到 `+0.5 m/s` 向前球速 **0/6**。因此不打开 fresh holdout，不晋升，不制作“球队已经会踢球”的视频。

## 可核查证据与步骤

- 协议：`docs/rsi/protocols/team-taskspace-family-v13b.json`、`docs/rsi/protocols/team-taskspace-course-sweep-v14.json`；合格 RoboNaldo G1 locomotion asset body hash `sha256:1525aafc953293dd340475e8660b58fae7ea1a22715222ab06331b644340626c`。六名 G1 共用一颗 MuJoCo 足球，每个 agent 仍有私有 ROSClaw cell 和动作边界。
- 机制实验：`/data/rosclaw_overflow/rsi-team-taskspace-family-v13b-corrected/report.json`，报告 hash `sha256:9e0f80d981ffba45b1386ab40199412fb2666097156581f2e2d0675b823f0f3c`。Parent：0 脚球接触；`lateral10`：0 脚接触，第 146–147 帧有非足部碰撞，因此安全门拒绝；`forward16`、`combined`：第 127–128 帧有红队组织者左脚真实接触，逐帧任务空间动作审计通过，全部身体安全，关节残差上限 0.35 rad。两种成功动作的物理轨迹基本相同，不能据此宣称侧向参数有独立贡献。
- 变体成对测试：`/data/rosclaw_overflow/rsi-team-taskspace-course-sweep-v14/report.json`，报告 hash `sha256:4d38e09570b6ccb772d4b6eb8bfbf385502e773547a3c16806eb701fd991858a`。`near/far/left/right/slow/fast` 六组各有冻结 Parent、Candidate、250 帧完整物理轨迹和动作轨迹。六组无脚球接触，无有效向前传球；Candidate 各组安全门均通过。最小足端中心到球中心距离分别约 `0.236/0.246/0.392/0.196/0.214/0.244 m`；说明命中域极窄。
- 每次试验均记录源码哈希、场景哈希、动作轨迹、物理轨迹和逐帧审计。所有训练/审计输出均在外部数据目录，不将大规模原始轨迹入 Git。

## 失败与加固

初次动作族实验被独立审计中断：实验电机把别人的球接触当成自己的触球而提前释放摆腿。修复为只接收 `agent_id` 相同且 `effector` 是左右脚的真实接触；增加回归测试。第二个问题是报告误按 fixture 原始顺序解释球员接触码，而共享世界按 `agent_id` 排序编码，曾把前锋触球误报为 0。修复后加入编号顺序单测，报告同时要求控制器子步观察与世界轨迹首个接触帧一致。失败轨迹保存在 `/data/rosclaw_overflow/rsi-team-taskspace-family-v13b/` 和 `/data/rosclaw_overflow/rsi-team-taskspace-family-v13b-attribution-fix/`，不覆盖或冒充新结果。

## 下一训练关口

问题已经从“有没有碰球”转为“如何对小扰动稳定触球并产生有用的方向/力度”。仅扩大关节补丁或拿单条成功轨迹剪片不会解决。下一阶段应把预触球**身体站位和步态相位**作为学习变量，保持 SONIC/RoboNaldo locomotion foundation 冻结，训练受限导航/足端策略；训练集含上述六类扰动和更大范围，目标同时约束足部接触、非足部拒绝、触球后球速与方向、G1 平衡。按 Parent/Candidate 成对评估，训练域先达到可重复的成功，才开启完全未见场景的 fresh holdout；若 holdout 不优于 Parent，拒绝晋升。只有通过该门并能接到传球、产生射门、和门将形成连续比赛，才值得做宣传片。
