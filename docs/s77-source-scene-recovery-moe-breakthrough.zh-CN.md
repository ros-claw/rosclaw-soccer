# S77：原足球场多姿态恢复 MoE 可达性突破

日期：2026-08-24
状态：**真实 post-save 开发快照 9/9；SIM_ONLY 可达性通过，尚不可晋升**

## 结论先行

本阶段第一次把门将真实扑救后的俯卧/侧卧状态，在 ROSClaw 原足球场 MuJoCo-Warp
场景中完整串成：

```text
真实 post-save 沉降状态
  → OpenTrack 多姿态 Get-up 专家
  → 本体感 + 足底接触连续交接门
  → 相位对齐 Capture 专家
  → recurrent locomotion 预热和渐进接管
  → 连续稳定 goalkeeper-ready
```

同一批 S49 的 9 个真实状态最终达到：

- 起身进入 ready 姿态：**9/9**；
- Capture→locomotion 交接完成：**9/9**；
- 最终连续稳定：**9/9（100%）**；
- 最短最终连续稳定时间：**21.04 秒**；
- 非有限状态：**0/9**；
- 自动重置、teleport、硬件命令：**0**。

这不是把成功阈值放宽，也不是播放动作文件。每个世界从保存的 36 维 qpos、35 维
qvel 开始，在原体育场碰撞模型里每 2 ms 计算并施加一次 29 关节力矩，最多连续运行
40 秒。最终门仍要求低线速度、低角速度和连续 2 秒稳定，并且 Capture 必须完成到
locomotion 的热交接。

但它仍是**特权可达性 oracle**：OpenTrack 专家读取参考动作相位，9 条入口路线曾使用
开发结果选择；本轮也还没有从新射门种子连续产生未见状态。因此不能声称统一的仅本体
小脑已学会，也不能晋升到真实 G1。

## 1. 为什么旧方法长期是 0/9

S49 的真实沉降语料是 8 个 PRONE、1 个 LEFT_SIDE，旧 RoboNaldo/MJLab R0 专家却来自
一条 SUPINE/侧仰卧起身轨迹。旧专家在自己的局部源分布上能成功，却没有门将落地所需
的接触拓扑。

本轮先做了两个严格反证：

| 路线 | 结果 | 解释 |
|---|---:|---|
| 真实高动量状态 → 本体相位对齐 MJLab | 0/9 | 动量和接触拓扑都不匹配 |
| 1 秒阻尼吸收 → 本体相位对齐 MJLab | 0/9 | 角速度已降至约 0.10–0.41 rad/s，仍缺 PRONE/SIDE 动作拓扑 |

因此问题不是“再给同一 residual 多训几轮”，而是缺少独立多姿态 Get-up 专家。这个
结论与 Athlete Foundation V1 的 Hybrid MoE 假设一致：错误接触拓扑必须换 expert，
不能靠小 residual 修补。

## 2. 引入 OpenTrack 多姿态教师

S51 已证明 OpenTrack `specialist2` 在自己的 MuJoCo 场景中可覆盖这 9 个状态，但当时
不能证明原足球场迁移。本轮新增内容绑定的 Torch runtime，直接解析固定 ONNX 图：

- 156 维参考条件 + 本体观察；
- 5 层 SiLU 隐层和 29 维 tanh 动作头；
- Torch 与 ONNXRuntime 随机输入最大误差小于 `2e-5`；
- 原始权重、配置、动作和恢复路线均由 SHA-256 绑定；
- 输出是 29 关节 PD 目标增量，实际物理子步只施加力矩。

首次迁移发现一个真实接口错误：MuJoCo free-joint 的旋转 qvel 已经位于子机体系，适配
器却又把它旋转了一次，等于向教师提供错误陀螺仪。修正后，原场景 OpenTrack 单专家从
8/9 暂时站起提升到 **9/9 进入 ready**，峰值角速度也下降；但它们仍无法连续稳定 2 秒。

这一步说明多姿态动作发现已迁移，失败转移到了终端稳定和技能交接。

## 3. 为什么第一次 MoE 只有 5/9

第一版在骨盆高度达到 0.62 m、upright projection 达到 0.75 并持续 0.2 秒后立刻切换
Capture。结果从单专家的 0/9 连续稳定提升到 **5/9**；5 个通过者随后稳定了
22.66–31.84 秒。

剩余 4 个都曾站起来，但切换时尚未建立可靠地面支撑。只看高度与姿态会把“身体刚好
掠过 upright 门、脚还没有承重”的瞬间误判为可交接状态。Capture 接到这种 successor
state 后无法补救，会再次跌倒。

这不是 Capture 专家本身不可达：它刚在内容绑定的 256 状态旧银行上实现 256/256。
真正问题是 predecessor 没有把状态送入 Capture 的可达域。

## 4. 最终通过的因果交接门

最终门只读取部署可见的当下本体感和接触，不读取未来结果：

```text
pelvis height >= 0.62 m
upright projection >= 0.75
root linear speed <= 0.75 m/s
root angular speed <= 2.0 rad/s
left foot OR right foot has physical support
all conditions continuously hold for 0.20 s
```

交接时再以关节位置/速度、骨盆高度/四元数因果选择 Capture 参考相位。Capture 进入
双足、低动量稳定包络后，先清空并预热 recurrent locomotion memory，再渐进混合控制权。

逐状态结果：

| 状态 | 姿态 | OpenTrack 入口 | 交接线速 m/s | 交接角速 rad/s | 足支撑 | Capture 相位 | 最终连续稳定 s |
|---:|---|---:|---:|---:|---|---:|---:|
| 0 | PRONE | 6230 | 0.574 | 1.050 | 双足 | 441 | 32.94 |
| 1 | PRONE | 1250 | 0.381 | 1.372 | 右足 | 442 | 27.72 |
| 2 | LEFT_SIDE | 7348 | 0.547 | 0.454 | 右足 | 449 | 21.50 |
| 3 | PRONE | 1246 | 0.387 | 1.429 | 右足 | 442 | 27.80 |
| 4 | PRONE | 1404 | 0.423 | 1.126 | 右足 | 371 | 30.68 |
| 5 | PRONE | 6232 | 0.374 | 0.706 | 双足 | 442 | 32.84 |
| 6 | PRONE | 7348 | 0.470 | 0.453 | 右足 | 378 | 21.60 |
| 7 | PRONE | 1360，2× 慢相位 | 0.457 | 0.903 | 右足 | 350 | 23.44 |
| 8 | PRONE | 7324 | 0.509 | 0.808 | 右足 | 428 | 21.04 |

这里的“最终连续稳定”不是首次站立到结束的总时长，而是通过低动量、双足和 locomotion
交接条件后，直到 40 秒考试终点仍连续满足宽松物理稳定条件的时间。

## 5. 对 ROSClaw 自进化架构的意义

这轮突破不是足球专用姿态补丁，而是证明了通用的 successor-state growth 机制：

1. **失败不是统一标量**：同一个“起身失败”可分成动作拓扑缺失、接口语义错误、终端
   稳定失败和 predecessor/successor 不可达。
2. **专家按物理职责成长**：OpenTrack 负责多姿态接触序列，Capture 负责卸掉站起末端
   动量，Athlete/locomotion 负责长期平衡；每个专家不再被迫同时解决所有问题。
3. **门控必须是可达性合同**：高度阈值不足，必须绑定速度、接触和连续保持。失败状态
   会反向修改 predecessor 的交接条件，而不是盲目修改后继专家。
4. **Growth 以成对证据升级假设**：0/9 → 9/9 ready → 5/9 full handoff → 9/9，
   每一步只改变一个可解释因素，并保留被拒绝证据。
5. **稳定性—可塑性分离**：已验证的 Capture 与 locomotion 保持冻结；新多姿态专家只在
   自己的职责域获得权力，未通过门的候选不会覆盖冠军。

同样的模式可用于抓取后的 manipulation-ready、碰撞后的 navigation-ready、跌倒后的
athlete-ready，不应把 Get-up、Capture 或 goalkeeper 名称硬编码进 ROSClaw Core。

## 6. 新增与加固内容

- `growth/opentrack_tracking.py`
  - 内容绑定 OpenTrack ONNX→Torch runtime；
  - exact 156-D observation contract；
  - local-frame gyro 语义修复；
  - 原始动作、配置和运动哈希校验。
- `training/opentrack_teacher_source_exam.py`
  - CPU 成功路线与真实快照逐状态绑定；
  - 原体育场 MuJoCo-Warp 无重置考试；
  - OpenTrack→Capture→locomotion MoE；
  - 本体/接触连续交接门；
  - 编译物理模型数值哈希、种子、GPU 和逐状态结果；
  - report hash validator，篡改或宣称漂移 fail-closed。
- `growth/mjlab_getup.py`
  - 允许 impact/前序技能在专家尚未启动前绑定因果参考相位；
  - 一旦专家启动，相位不可再修改。
- `training/recovery_snapshot_exam.py`
  - 冲击吸收后相位选择的严格反证考试；
  - 明确保存 0/9 负结果，避免误把卸力当完整恢复。

## 7. 证据

最终通过证据：

```text
/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/
  opentrack-capture-moe-source-stadium-v3.json
```

- report hash：`sha256:14f4f33269c02abce9dd36ee071c6ab329ca859be83d2bbdf807a65699ffca4e`；
- 文件 SHA-256：`33e3c1faff7402db0c2e1f7ea4e27bfb33b21ece2d18b2028616a5a4eb8ada47`；
- 编译场景 hash：`sha256:165e898779c29924a5959356c5f6aad40e35e0f6e8069d24cbe66ef49bf75497`；
- GPU：`cuda:0`，NVIDIA RTX A6000；
- 固定随机种子：`76101`；
- 物理后端：`mujoco_warp_source_stadium_no_auto_reset`。

重要被拒绝证据也保留：

- `s49-phase-aligned-direct-v1.json`：直接相位对齐 0/9；
- `s49-absorb-phase-aligned-v1.json`：吸收后相位对齐 0/9；
- `opentrack-teacher-source-stadium-v1.json`：错误 gyro 语义，8/9 ready、0/9 stable；
- `opentrack-teacher-source-stadium-v2.json`：gyro 修复，9/9 ready、0/9 stable；
- `opentrack-capture-moe-source-stadium-v1.json`：仅姿态交接，5/9 stable；
- `opentrack-capture-moe-source-stadium-v2.json`：本体/接触交接，9/9 stable；
- `opentrack-capture-moe-source-stadium-v3.json`：同结果的完整内容绑定正式证据。

### 7.1 轨迹绑定复演与视频

为避免“视频好看但数值考试不是同一次运行”，另跑了一次同配置、同 9 状态的正式考试，
在控制环内仅以 25 Hz 旁路保存每一帧 `qpos`。该次考试仍是 **9/9**，轨迹 archive 与
考试 report 双向绑定：

```text
/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/
  opentrack-capture-moe-source-stadium-v4.json
  opentrack-capture-moe-source-stadium-v4-trajectories.npz
```

- v4 report hash：`sha256:519dc7a15b8f764bf4e89749fd47df972fed97b97963dd0f0a359f96029ce87d`；
- v4 文件 SHA-256：`9d3a013c5cc032dd7e69b3cefe6b949e976500f67ea8545b1d480c12ce400d49`；
- 轨迹 SHA-256：`be0980053363a471cfbc2b9a6fabe8bfeefcfc63224864cbcfb707c44562b2dd`。

从 v4 的 9 个通过状态中选取 PRONE、LEFT_SIDE 和不同入口路线的 4 段，仅复演已经计分
的 qpos，不重算控制、不用像素判定成功：

```text
/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/showcase/
  recovery-moe-source-stadium-v1.mp4
  recovery-moe-source-stadium-v1.json
```

- 81.44 秒，1920×1080，25 fps，H.264，共 2036 帧；
- 视频 SHA-256：`31e81e1f97f8d2c09324a2d92c29f1cbfe6ce728a5ac8325df4519003c5f0870`；
- manifest 自哈希：`sha256:6af7e72505b10fd607a338d2f10a787b32d402de9a02b2f60f4e1678335ad0f3`；
- manifest 文件 SHA-256：`a2f7f023bf9085dff0f80111180568f1204e6a881a6c636c1296a24d7b53c71b`；
- 画面明确标注 Get-up、Capture + handoff、Locomotion ready；
- `visualization_only=true`、`pixels_used_for_scoring=false`、`promotion_eligible=false`。

视频 manifest 新增 fail-closed 校验：manifest 任一字段漂移或 MP4 内容哈希不一致都会拒绝。
逐段抽帧目检确认状态、路线和阶段标签可读，未发现黑帧、分辨率漂移或段落错位。

## 8. 不能越过的结论边界

当前可以说：

> 在 ROSClaw 原足球场 MuJoCo-Warp 场景中，一个职责分离的特权多专家链，对 9 个真实
> post-save 开发沉降状态实现了 9/9 起身、Capture→locomotion 交接与最终连续稳定，
> 证明这批状态在正确的动作拓扑和 successor gate 下物理可达。

当前不能说：

- 统一、仅本体感的恢复小脑已经学会；
- 新射门种子的未见 post-dive 分布已经 100%；
- 完整 save→land→recover→second-save 连续链已晋升；
- 该 oracle 可以部署到真实 G1；
- 9 个开发状态代表所有左右侧卧、仰卧、不同摩擦和高冲量情况。

所有代码和证据保持 `SIM_ONLY`，`hardware_authorized=false`，硬件命令为 0。

## 9. 下一阶段硬门

1. 从未参与路线选择的新射门种子采集真实 `POST_SAVE_FLIGHT/IMPACT/SETTLED`，形成独立
   sealed source-scene holdout；至少覆盖 prone、supine、left/right side 和不同支撑拓扑。
2. 先固定当前 oracle，对未见状态只允许因果入口匹配，禁止按物理结果换路线；报告
   posture 分层成功率和 95% 置信下界。
3. 记录 oracle rollout 的本体历史、接触历史和动作，训练统一 recurrent student；训练
   critic 可读特权状态，actor 禁止 reference phase、route ID 和未来结果。
4. student 必须 on-policy 物理训练，失败优先回流；不能再用离线 MAE 代替闭环成功。
5. 通过未见快照后接回连续剧集：扑救→受控落地→恢复→goalkeeper-ready→新横移/二次球，
   全程无 reset/teleport。
6. 宣传视频只从完整通过剧集中选择，并叠加真实事件、速度、接触和交接状态；视频不参与
   数值判定。

## 10. 最终质量门

- `ruff`：全部相关源码与测试通过；
- `mypy --follow-imports=skip`：9 个相关源码文件通过，0 error；
- `compileall`：相关源码通过；
- 两套依赖环境的聚焦回归：**22 passed**；
- 视频编码/尺寸/帧率、视频 SHA-256、轨迹 SHA-256、考试 report hash 和 manifest
  自哈希均已复核。
