# S115：重球全身运动课程与邻域稳健性闭环

## 结论先行

S115 把 S114 未通过的 `0.46 kg / 摩擦 0.16` 重球密封留出，正式转成了第二前锋的全身运动课程。我们没有提高脚部“球炮”外力，也没有用视频像素评分，而是将足部俯仰姿态作为可审计、可复现的运动上下文，与失败更新后的接触 actor 一起进入原有四台 G1、两颗实体球、传球—射门—扑救—恢复的连续 MuJoCo 世界。

结果有进步，但不能夸大：`0.1261 rad` 控制点连续两次严格回放通过，重球被右脚真实接触后在 `1.373 m` 高度被门将手套真实扑出，候选 actor 接管、冻结父策略未接管，前锋未摔倒，门将最终恢复 ready。可是仅把足部俯仰改为 `0.12605 rad`，相邻密封留出虽然仍在 `1.424 m` 发生真实手套接触，却未能在终场满足双脚支撑、低线速度和低角速度门。因此资格组合明确返回 `REJECTED_SEALED_HOLDOUT_TASK_FAILURE`，不允许广泛晋升。

这正是 ROSClaw 自进化应有的闭环：发现单点突破，同时拒绝把脆弱单点包装成成熟能力，并把失败边界保存为下一轮数据。

## 本轮开发

### 1. 独立绑定全身运动上下文

`RoleIsolatedSecondStrikerProbeConfig` 新增可选的 `second_striker_foot_pitch_offset_rad`：

- 有限值且范围限制为 `[-0.18, 0.18] rad`；
- 写入请求、配置哈希和可复现闭包；
- 只作用于第二前锋自身的 29 DoF 物理控制配置；
- 仍然硬顶 `SIM_ONLY`，不发送硬件命令。

球质量、地面摩擦、身体姿态和接触 actor 因而成为四个独立且可验证的实验变量，后续训练可以学习上下文策略，而不必继续堆叠场景脚本。

### 2. 控制/留出资格门绑定上下文

资格器现在分别绑定控制和密封留出的：

- 球质量；
- 地面摩擦；
- 足部俯仰上下文；
- 候选 actor 内容哈希。

候选 actor 可以位于不同证据目录，但字节必须与失败更新产物完全一致。这既允许重绑定新证据闭包，也阻止拿另一个策略冒充同一候选。

### 3. 失败边界搜索

初步搜索表明，简单增加教师脚部外力不能稳定抬起重球；激进提高全身速度、俯仰和挑射幅度又会破坏整场安全。单变量课程最终定位到很窄的接触/恢复边界：

| 足部俯仰 | 结果 |
|---:|---|
| 0.1200 rad | 手套接触约 0.826 m，高度和终态不足 |
| 0.1250 rad | 球到达门将区域约 1.187 m，但无手套接触 |
| 0.1261 rad | 完整链路通过，手套接触 1.373 m，终态 ready |
| 0.12605 rad | 手套接触 1.424 m，但终态恢复失败 |
| 0.1275 rad | 进球并破坏整场安全门 |

这些结果说明当前动力学不是一个可用常数偏置覆盖的平滑区间。只有学习“触球前全身状态—触球冲量—触球后恢复”的联合分布，才可能得到真正稳健的小脑能力。

## 正式证据

### 失败记忆重绑定

- [evidence.json](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s115-contact-failure-memory-rebind-v3/evidence.json)
- report hash：`sha256:8c9c564f0a6b962be62bd2b19b302170c4f87d5937500bc30bcb20afb482a2b2`
- 更新后的 actor 字节仍与 S114 相同；这里只重建了当前源码闭包，没有偷偷重训或改权重。

### 重球控制点（两次严格回放）

- [evidence.json](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s115-heavy-pitch-control-v4/evidence.json)
- 上下文：质量 `0.46 kg`，摩擦 `0.16`，足部俯仰 `0.1261 rad`
- 状态：`QUALIFIED_DEVELOPMENT_CANDIDATE`
- 第二次真实手套接触高度：`1.373278 m`
- 第二前锋接触力峰值：`831.682 N`
- 候选接管帧：`1`；冻结父策略接管帧：`0`
- 前锋触球后摔倒：`false`
- 终态：八个 ready 子门全部通过
- report hash：`sha256:ebe62f77f88158e2cd0da4bba77e58c7e2f55f4dc672de69aa1c89f6d8f9ebf0`

### 相邻密封留出（两次严格回放）

- [evidence.json](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s115-heavy-pitch-neighbor-holdout-v4/evidence.json)
- 上下文仅把足部俯仰改为 `0.12605 rad`
- 状态：`REJECTED_TASK_FAILURE`
- 第二次真实手套接触高度：`1.424421 m`
- 前锋仍未摔倒；但是门将终态双脚支撑比例只有 `0.38`，根部最大线速度 `0.325 m/s`、角速度 `1.265 rad/s`，终态 ready 未通过
- report hash：`sha256:4920d4fe41bfa98c9ff5acb6dbcfb01b7f60bb23316e45e6a16c74dc8b6082b0`

### 组合资格结论

- [qualification evidence](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s115-heavy-pitch-qualification-v5/evidence.json)
- 控制完整链路：通过
- 相邻留出完整链路：失败
- 最终状态：`REJECTED_SEALED_HOLDOUT_TASK_FAILURE`
- report hash：`sha256:6e73993524a75ffd34e88890006a8ccab856e3a958eb82d37cef17c63f2e38bc`

## 可视化

- [s115-heavy-ball-control.mp4](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s115-heavy-pitch-control-video-v2/s115-heavy-ball-control.mp4)
- [视频清单](/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s115-heavy-pitch-control-video-v2/s115-heavy-ball-control.json)
- 1920 × 1080，30 fps，1069 帧，35.633 秒
- 视频 SHA-256：`d413a7f20a11de8b86ef90f1b49f6f8d6f8e98f6f2e57fee06cc2d0d41e37d6e`

视频只是对物理证据轨迹的渲染。成功/失败、手套接触、是否进球、是否摔倒和最终 ready 均来自仿真状态与接触传感器，像素从不参与评分。

## 软件验证

- Ruff 规则检查：通过；
- mypy：233 个源码文件通过；
- S115 定向测试：13 个通过；
- 全量测试：713 个通过、15 个因可选资产/栈缺失跳过；
- 另有 8 个历史外部证据测试因当前实现哈希变化按设计失效，显式排除这些已知历史节点后仍为 713 个全部通过。本轮没有新增功能回归，也没有重写历史证据来掩盖闭包变化。

## 对 ROSClaw 通用成长架构的意义

足球只是压力测试。S115 验证的是更通用的能力：

1. **上下文显式化**：把环境参数和身体参数从隐含脚本提升为候选策略的输入契约。
2. **失败驱动课程**：密封留出失败自动定义下一轮课程，而不是人工宣布“训练成功”。
3. **稳定性—可塑性门控**：控制点成功证明可塑性，邻域失败阻止脆弱能力污染冻结冠军。
4. **因果证据绑定**：actor、上下文、源码、依赖、进程契约和轨迹全部进入内容哈希闭包。
5. **角色局部成长、世界整体考试**：只允许第二前锋学习，但晋升仍由传球者、第一前锋、门将和两颗球的连续终态共同决定。

这套模式可以迁移到机械臂负载变化、移动机器人地面摩擦变化、无人机风场变化等任务，而不应固化为 G1 足球专用逻辑。

## 下一轮：从参数点搜索升级为稳健小脑策略

S116 不应继续寻找第四位小数的“幸运角度”。计划是：

1. 收集俯仰上下文、足端速度、支撑脚压力、骨盆姿态、球速与恢复状态的边界样本；
2. 训练低维残差 actor 同时输出触球前姿态残差和触球后恢复参数，而不是直接放大脚部外力；
3. 采用域随机化覆盖球质量、摩擦、初始球位和接触相位，并以 CVaR/最差分位数而非均值选优；
4. 将门将终态恢复损失反向归因到射门轨迹，防止“扑到球但把队友留在失稳状态”被记为完整成功；
5. 冻结当前 S114 冠军，只有跨多个未见邻域、严格重放、全世界安全与终态门同时通过时才晋升。

当前正确结论是：**重球单点已有突破，稳健重球技能尚未学会，因此不晋升。**
