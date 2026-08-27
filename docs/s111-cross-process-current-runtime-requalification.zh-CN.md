# S111：跨进程当前运行栈重认证与完整四 G1 链恢复

日期：2026-08-27
边界：`SIM_ONLY`，CPU MuJoCo 物理裁判；没有 ROS/DDS/真实 G1 指令；视频像素不参与评分。

## 结论先行

S111 没有把一次偶然成功或手工加力包装成“自进化”。本轮先复现 S110 的失败，再发现旧证据边界仍不完整，最终得到以下结论：

- S110 的正式 `right-control` 轨迹失败，但其证据只绑定了选定实现文件、ROSClaw Core 的一个文件和数值库版本，没有绑定传递依赖源码树与进程执行契约；
- 在当前 `main@f8249c4` 和当前资产下，冻结的旧控制器无需运动参数补丁即可通过完整链；
- 3 个全新 Python 解释器、3 个不同 PID 分别从时间零运行 4 台 G1、2 颗实体球和 25 秒 MuJoCo 世界；
- 3 次结果对象、评价对象、trajectory digest 与压缩 NPZ 字节完全相同；
- 每次都通过首扑、落地、rearm、第二前锋右脚触球、二扑、世界安全和最终 ready 的全部门控；
- 当前冠军晋升为 `PROMOTED_SIM_ONLY_CURRENT_RUNTIME_FULL_CHAIN`，但不产生真实机器人授权；
- 开发中找到的 58 N / 0.60 垂直拳击候选虽然恢复了首扑，却破坏了第二扑，已按 fail-closed 淘汰，没有进入正式配置。

这不是“把门槛改低后通过”。S111 复用 S110 的 `right-control` 质量、摩擦、站位、第二前锋和全部评分阈值；变化的是证据闭包和重启级复演方式。

## 为什么 S110 与当前结果不同

### 已证实

1. S110 保存的 `right-control` 物理轨迹确实失败，不能篡改历史结论；
2. 当前源码和当前资产在同一进程内双跑、四个独立并行进程以及 S111 三个正式顺序进程中，都稳定得到同一个成功摘要 `sha256:c6916841…`；
3. S110 的 `implementation_hash` 只覆盖若干直接文件，没有覆盖 recovery integration、配置装配、媒体之外的完整足球 Python 源码树；
4. S110 只记录 MuJoCo/NumPy/ONNX Runtime 版本，没有记录 Python 实现、平台、libc、CPU 数、哈希随机化开关和数值线程环境；
5. 因此，S110 证据不足以证明“同一传递实现闭包、同一进程契约”下发生了退化。

### 没有伪造的结论

现有证据不能唯一定位到某一行历史代码、某个线程变量或某次外部状态变化，所以本报告不声称已经证明唯一根因。S111 的工程修复是把这些过去未承诺的状态全部纳入内容绑定，并要求跨解释器完全一致；以后只要完整源码树、资产、运行库或进程契约变化，就必须重新认证。

## 开发内容

### 1. 新增跨进程认证器

`current_runtime_prefix_requalification.py` 现在会：

1. 读取并验证 S110 被拒绝的 population evidence；
2. 精确重建其 `right-control`：`left-inner`、右脚、0.41 kg 球、0.10 草地摩擦和 25 秒连续世界；
3. 为每次 replay 启动全新的 Python 解释器，而不是在同一个进程里重复调用函数；
4. 每个 worker 独立保存 result/evaluation、trajectory NPZ、物理摘要、PID 和进程契约；
5. 主认证器逐字比较跨进程评价对象与摘要，并验证每个 worker/trajectory 的内容哈希；
6. 独立 validator 会重新读取 NPZ、重算 trajectory digest、重建十项资格门，而不是信任 evidence 预填的“通过”；
7. 任一 worker 失败、PID 不独立、进程契约不同、轨迹不同、路径逃逸或证据被改写都会 fail closed。

### 2. 从“选定文件哈希”升级为“完整源码树哈希”

request 现在绑定：

- `rosclaw_soccer` 全部 Python 源码树；
- 外部 ROSClaw Core 全部 Python 源码树；
- Git commit；
- G1 Body/Kick Prior；
- 前锋 actor、门将 actor、GMT 模型/skill、Dive Athlete、Recovery Athlete 及其考试；
- MuJoCo、NumPy、ONNX Runtime 版本；
- Python 版本/实现、操作系统、机器架构、libc、CPU 数、哈希随机化和线程环境变量；
- S110 前驱 evidence 的文件哈希与 report hash。

这个闭包偏保守：无关 Python 文件变化也会触发重新认证，但比把跨版本差异误认为“机器人学会/忘记”更安全。

### 3. 失败候选不覆盖当前冠军

开发搜索曾找到一组首扑通过候选：58 N 基础双臂拳击、0.60 垂直分量、0.78 下肢模仿幅度。它在完整四 G1 世界中得到：

- 首扑和落地全门通过；
- rearm、第二前锋触球和高球发射通过；
- 二次手套接触/二扑解围失败。

随后同一正式进程中的冻结旧控制器反而通过了两次扑救全链，因此候选被判为负迁移。S111 没有把它写入运行默认值，体现 Stability–Plasticity 的基本原则：可塑候选只有在完整后继任务不退化时才允许替代稳定记忆。

### 4. 新增证据下游视频

`current_runtime_requalification_video.py` 从正式 worker trajectory 渲染 4 台 G1、2 颗球和两次扑救。视频 manifest 绑定 evidence、request、trajectory 和 MP4 字节；视频永久声明：

- `SIM_ONLY`；
- 3 次跨进程严格复演；
- 无球炮、无 reset、无 teleport；
- 像素不参与评分；
- 视频本身不具备 promotion authority。

## 正式物理结果

| 事件 | 正式值 |
|---|---:|
| 传球触球 | 5.602 s |
| 第一前锋触球 | 7.390 s |
| 第一次手套接触 | 8.014 s / 1.417 m |
| 双脚离地 | 0.160 s |
| 起跳峰值竖直速度 | 1.115 m/s |
| 落地竖直速度 | 0.673 m/s |
| 落地角速度 | 1.653 rad/s（门限 3.5） |
| 门将 rearm | 16.000 s |
| 第二前锋右脚触球 | 16.948 s / 791.224 N |
| 第二球触后峰值 | 10.158 m/s |
| 第二球前向峰值 | 8.816 m/s |
| 第二次手套接触 | 17.512 s / 1.441 m |
| 二扑后最小前向速度 | -1.033 m/s |
| 二扑后峰值横向解围速度 | 4.678 m/s |
| 最终 ready 根线速度上限 | 0.000253 m/s |
| 机器人互撞 / 关节 / 力矩 / 饱和 | 0 / 无 / 无 / 无 |

全部 14 个连续链门控为真，包括：

- 第一扑是接触/腾空/落地全部合格的真实扑救；
- 4 台 G1 和 2 颗球从时间零存在；
- 门将在第二前锋触球前实测 ready/rearm；
- 第二球触球前峰值仅 0.000518 m/s；
- 学习 actor、目标 muscle memory、torque muscle memory 均实际激活；
- 新的门将飞行 epoch 因第二前锋触球因果触发；
- 第二次手套接触面距离 -12.205 mm，在碰撞门限内；
- 第二球没有入门，且被真实反向/横向清出；
- 第二前锋稳定、全世界安全、控制时钟连续、门将最终双支撑 ready。

## 证据与视频

正式证据：

- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s111-cross-process-current-runtime-requalification-v4/evidence.json`
- 同目录 `request.json`、3 个 worker JSON 和 3 个 trajectory NPZ；3 个 NPZ 的 SHA-256 均为 `4638c1ec…`。
- evidence 文件 SHA-256 为 `41ccb494…`，内部 report hash 为 `cab7866c…`。

正式视频：

- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s111-cross-process-champion-video-v2/s111-cross-process-full-chain.mp4`
- 40.267 秒，1920×1080，30 fps，1208 帧；视频 SHA-256 为 `ddfcb676…`。

被拒绝的开发证据保留用于审计，不可作为冠军：

- `s111-current-runtime-prefix-requalification-v1`：错误假设独立三 G1 前缀会复现 S110 退化；
- `s111-current-runtime-prefix-requalification-v2`：58 N 候选首扑通过但二扑退化。

## 通俗解释

S110 像一本实验记录：当时同一套战术输球了，而且连续回放两遍都输。问题是它只抄了“MuJoCo 版本”和几张关键战术纸，没有把整个教练组手册、解释器环境和后台线程方式都封存，所以后来无法证明两次实验真的处于完全相同的执行世界。

S111 先想给门将加力，结果第一球扑住了，第二球却漏了——这是典型的“学新忘旧”。于是系统拒绝这段新肌肉记忆，重新测试旧冠军。旧冠军在多个完全重启的进程里，不但第一球扑住，恢复后还扑住第二前锋的高球，而且每一个数值字节都一致。

真正的成长不是“这次看起来赢了”，而是：知道什么时候不该改参数，知道旧证据缺了什么，补齐完整可复现边界，并让当前稳定记忆通过更严格的考试。

## S112 下一阶段

1. 把跨进程/完整源码树 evidence closure 下沉为通用 ROSClaw SimForge 能力，而非足球专用约定；
2. 在 S111 冻结冠军上做一角色可塑的第二扑 holdout population，不再改首扑冠军；
3. 加入进程重启、CPU 负载与线程环境的显式扰动种群，验证是否仍为字节级一致；
4. 扩展到左脚和镜像 lane 前，先修复初始拓扑可达性和机器人安全距离；
5. 所有候选必须通过“当前全链不退化 + 跨进程 replay + 未见 holdout”三重门，才能进入宣传冠军。
