# S112：把足球复演经验下沉为 ROSClaw Core 通用闭环

日期：2026-08-27  
边界：`SIM_ONLY`；CPU MuJoCo 物理裁判；没有 ROS、DDS、串口、CAN、厂商 SDK 或真实 G1 指令；视频像素不参与晋升。

## 结论先行

S112 的突破不是给门将再加一个足球补丁，而是把 S111 暴露出的“同一结果到底能不能相信”变成 ROSClaw Core 的任务无关能力，并让足球案例真正改用该能力：

- ROSClaw Core 新增通用 `simforge.reproducibility`，PR #475 在全量 CI 和 Data Flywheel gate 全绿后合并到 `main@5ebeb7f3`；
- Core 能绑定任意任务的源码树、运行依赖、模型/数据/前驱证据、Python/平台/数值线程契约，并统一推导 fresh PID、闭包一致、严格复演、结果通过和 `SIM_ONLY` 安全门；
- Soccer 不再自己把 `fresh_process_count=true` 等字段当成事实，而是把每个 worker 的原始报告交给 Core 重新判定；
- 每个 worker 在运行 MuJoCo 之前都独立重建闭包，闭包不一致直接停止，不允许“父进程绑定、子进程只回声”；
- 正式 launcher PID 为 `1454499`，3 个新 worker PID 为 `1454580 / 1481803 / 1508855`；worker 不复用 launcher，三份评价对象、trajectory digest 和压缩 NPZ 字节完全一致；
- 物理结果没有因证据架构重构发生漂移：NPZ 仍为 `sha256:4638c1ec…`，trajectory digest 仍为 `sha256:c6916841…`；
- 13 项最终资格门全部通过，其中 7 项由 Core 通用 verdict 推导，6 项由 Soccer 的前驱、物理和权限语义推导；
- 结果仍只是单一 `left-inner/right-control` 的研究冠军，不代表双侧、随机 holdout、真实机器人或商业授权。

## 1. ROSClaw Core 开发

Core 新模块 `rosclaw.simforge.reproducibility` 完全不知道 G1、足球、门将或 MuJoCo 评分规则，提供：

1. 有限、类型保持、稳定排序的 canonical JSON SHA-256；
2. 大文件流式哈希，拒绝文件和源码树中的符号链接；
3. 只按相对路径与内容绑定、与 checkout 绝对路径无关的源码树摘要；
4. 模型、策略、数据和前驱证据的文件摘要与字节数；
5. Python 实现/版本、OS、机器架构、libc、CPU 数、hash randomization 和数值线程环境合同；
6. `ReproducibilityClosure` 及稳定 closure hash；
7. `evaluate_cross_process_replays`，从原始 worker 报告重新推导 7 个通用门；
8. 永久 `SIM_ONLY`、不能授予硬件权限的 fail-closed verdict。

Core 定向测试为 16 passed；完整 SimForge 为 188 passed、1 skipped、3 deselected；Practice 为 194 passed、4 skipped。GitHub 上 Python 3.11/3.12/3.13、Lint、Type Check、30 分 39 秒 Full Regression、39 分 10 秒 Data Flywheel gate、ROS Docker、Hub、Evidence Pack、跨 UID 和产品验收全部通过。

## 2. Soccer 如何消费通用能力

Soccer 的认证请求现在包含 Core 原生 closure 和 closure hash。正式闭包绑定：

| 类型 | 正式范围 |
| --- | --- |
| Soccer 源码 | 228 个 Python 文件 |
| ROSClaw 运行栈 | 902 个 Python 文件 |
| Core SimForge 可复现实现 | 97 个 Python 文件 |
| 外部门将参考源码 | 34 个 Python 文件 |
| 文件工件 | 14 个 |
| 数值依赖 | MuJoCo、NumPy、ONNX Runtime |
| 独立复演 | 3 个新解释器/PID |

14 个工件包括前锋/门将 actor、GMT model/skill、Dive Athlete checkpoint/exam、Recovery Athlete checkpoint/exam、RoboNaldo G1 模型/场景/运动/策略/FreeKick 源和 S110 前驱 evidence。

路径只用于外部证据复验时重新定位文件；closure hash 本身只承诺标签、相对源码路径、内容、版本和进程合同，因此搬动 checkout 不会凭空改变内容身份。validator 会重新打开 14 个工件和 4 棵源码树，而不是只核对 request 自己的哈希。

## 3. 正式通用门控

Core verdict 的 7 项全部为真：

| Core gate | 含义 |
| --- | --- |
| `expected_worker_count` | worker 数严格等于请求的 3 |
| `fresh_process_identity` | PID 唯一，且不复用 launcher PID |
| `process_contract_identical` | Python/平台/线程合同完全一致 |
| `closure_bound` | 每个 worker 都绑定同一 Core closure hash |
| `cross_process_exact_replay` | evaluation、digest、NPZ hash 完全相同 |
| `worker_outcomes_passed` | 每次任务评价都通过 |
| `all_workers_sim_only_safe` | 无硬件授权、无硬件命令，均为 `SIM_ONLY` |

Soccer 另外重新计算前驱确实被拒绝、首扑 contact/takeoff/landing 通过、完整连续球队链通过、四棵源码树齐全、当前闭包一致和最终权限上限。最终不是把 Core 的 `passed` 再抄一遍，而是 Core 通用门与 Soccer 物理门的交集。

## 4. 正式物理结果

| 事件 | 正式值 |
| --- | ---: |
| 传球触球 | 5.602 s |
| 第一前锋触球 | 7.390 s |
| 第一次手套接触 | 8.014 s / 1.417 m |
| 双脚离地 | 0.160 s |
| 落地角速度 | 1.653 rad/s（门限 3.5） |
| 门将 rearm | 16.000 s |
| 第二前锋右脚触球 | 16.948 s / 791.224 N |
| 第二次手套接触 | 17.512 s / 1.441 m |
| 最终 ready 根线速度上限 | 0.000253 m/s |
| 机器人互撞/关节/力矩/饱和 | 0 / 无 / 无 / 无 |

这说明证据架构升级没有暗改动作参数，也没有通过降低门槛换取成功。S111 的同一冻结肌肉记忆在更严格的 Core 闭包下仍得到同一物理轨迹。

## 5. 正式证据与阶段视频

证据：

- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s112-core-closure-current-runtime-requalification-v4/evidence.json`
- evidence 文件 SHA-256：`1ec81822e0b80016c327841af5318c0a904a0feefc2c602539ca407425461371`
- report hash：`sha256:ab43ab5dcf1846c8b1cfef138077793ca737727b7fe7eae090841c22dffa4e8a`
- closure hash：`sha256:224f74ff09c2180afc569a8bbcf45256f83353291dcc6dd75bf7e9ae6ec84597`
- Core verdict hash：`sha256:9e6e8c54389c6cddb062990e6ecb6fe755004d989aeb086a1f1504539be2a978`

视频：

- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s112-core-closure-champion-video-v3/s112-core-closed-full-chain.mp4`
- 40.267 秒、1920×1080、30 fps、1208 帧、H.264；
- 视频 SHA-256：`f1b05337b11abd3a4d7918544903626c288bd6cbcf95cbe9d4df40c9c00dd5f6`；
- manifest hash：`sha256:417448298219a9c3015ac1c787ed632f6552138730d685965a077f7c0be33f65`；
- manifest 绑定正式 evidence、request、trajectory 和 MP4；视频仍是证据下游展示，不参与晋升。

## 6. 通俗解释

S111 相当于足球队自己制定了一套“录像防伪规则”，虽然规则已经严格，但别的机器人任务不能复用，球队也可能同时当运动员、裁判和公证员。

S112 把公证规则搬到了 ROSClaw 总部：任何任务都能把自己的教材、教练代码、肌肉记忆、运行环境和考试产物装进一个有内容指纹的密封箱。三个互不相干的新进程先检查密封箱，再各踢完整比赛；总部只负责判断“是不是同一批材料、是不是新进程、结果是否逐字一致、有没有越过仿真权限”，足球裁判只负责判断“球到底有没有被脚踢、门将到底有没有接触、机器人是否落地和恢复”。两个裁判都通过，才叫可复现冠军。

它没有让本轮门将动作更花哨，但让以后任何自进化都更可信：模型变强、变弱或遗忘时，我们能够区分真实学习变化和环境/代码漂移。

## 7. 仍未解决与 S113

1. 当前是严格同机 CPU replay，不是跨机器、跨 CPU 或容差统计复演；
2. hash 是内容承诺，不是签名，未来需把 closure/verdict 接入 ROSClaw 签名 evidence pack；
3. 当前只认证冻结右侧单 lane，没有证明新候选能在未见 lane 上成长；
4. 下一阶段应冻结首扑冠军，只开放第二扑/恢复角色的有限可塑区，用未见 holdout + 当前全链 retention + Core 跨进程闭包三重门晋升；
5. 负迁移必须作为 Growth 经验保留，不能只留下成功宣传片。
