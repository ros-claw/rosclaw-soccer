# S49：从真实扑救状态学习完整恢复链路

## 结论先行

完整链路不能靠把“扑救策略、起身模型、站立小脑”三个现成模块顺序调用来学好。
真正的学习对象是模块之间的状态分布：扑救触球后，G1 常以很低的骨盆高度、较大
线速度和角速度进入地面接触；而 S48 的神经起身专家只在一条固定起身轨迹及其局部
扰动附近通过了考试。两者之间存在明显的分布断层。

本阶段建立了可复用的 ROSClaw 闭环：从真实前序技能保存物理快照，进入不可变失败
记忆，按动量和姿态建立课程，用外部动作数据作为训练教师而不是直接控制器，最后只
允许通过完整连续剧集考试的候选晋升。它不是足球专用补丁；抓取后复位、碰撞脱困和
跌倒恢复都可复用同一套 `post-skill recovery` 协议。

本轮也得到一个诚实的负结果：现有 RoboNaldo/MJLab 起身专家在 9 个真实扑救失败
状态上是 **0/9**，加 1 秒物理冲量吸收后仍为 **0/9**。因此没有把它接入正式守门
策略，也没有制作会掩盖失败的视频。当前突破是找到了正确的数据边界、修复了错误的
MotionDecode 起身片段，并建立了下一轮真正可以训练的课程入口；完整链路尚未晋升。

## 一、为什么以前“每个模块都能跑”，串起来仍然失败

S48 在源倒地姿态及小扰动上实现了 28/28 起身、热交接和最终站稳，但这只证明一个
局部组件。真实扑救策略产生的 9 个失败终态具有：

- 骨盆高度 0.100–0.119 m；
- 根线速度 0.343–2.158 m/s；
- 根角速度 2.577–8.558 rad/s；
- 机器人仍在滑动、翻滚或二次碰撞，而不是已经静止的标准俯卧/仰卧姿态。

直接运行冻结起身专家后，9 个状态均未进入交接包络，最终稳定 0/9，且峰值角速度
达到 8.81–12.58 rad/s。原因不是“RL 训练轮数还不够”，而是输入状态不属于该专家
的训练分布。强行接入会让守门链路更差。

用 1 秒关节阻尼吸收冲量后，9 个状态的角速度降到 0.088–0.403 rad/s，证明冲量
吸收本身可以单独解决；但起身仍是 0/9，说明问题还包括姿态和接触模式不匹配。
因此恢复必须拆成两个连续但不同的控制阶段：

1. 冲量吸收：在接触中卸掉线/角动量，避免头部冲击和关节限位；
2. 多姿态起身：从实际沉降后的侧卧、俯卧、仰卧和支撑姿态回到双足低动量站立。

之后才进入 S48 已验证过的终端保持、locomotion 预热和渐进权限交接。

## 二、已有数据和框架怎样分工

| 数据或框架 | 本地可用内容 | 正确用途 | 不能据此声称什么 |
|---|---:|---|---|
| MotionDecode | 坐到站 91、俯卧到站 194、躺到站 182、跌倒/恢复 549，共 1016 条 | 多姿态动作形状、自然度、时间相位教师 | 不是物理合格控制器，也不代表真实扑救落地分布 |
| MOSAIC | 29 条原生 G1 stand-up NPZ | 增加 G1 机体上的起身动作多样性 | 旧 GMT 直接迁移失败，不能直接部署 |
| G1-retargeted LAFAN | 6 条 `fallAndGetUp` | 少量恢复专家种子和对照 | 覆盖不了真实冲量与所有姿态 |
| RoboNaldo/MJLab | 1 个已通过局部物理考试的起身专家 | 局部专家、终端保持与交接基线 | 不能泛化到本轮真实扑救失败状态 |
| ProtoMotions | recovery reset 机制 | 借鉴恢复课程和重置流程 | 随机造倒地姿态不能替代真实链路快照 |
| OpenTrack | 多专家 DAgger 工具 | 把姿态路由专家蒸馏成只看本体感的单一学生 | 蒸馏前仍需先训练出物理合格的各姿态专家 |

可选的分布式训练框架（例如 AReaL）适合做任务调度、rollout、检查点和资源编排，
但不决定关节策略的语义；它不能替代上述状态协议、奖励、教师和晋升考试。

## 三、本阶段已经实现的通用闭环

### 1. Practice：采集真实技能边界

新增 pickle-free 的恢复快照协议，保存 36 维机器人 qpos、35 维 qvel、实际动作、
双足接触、球与目标、事件阶段，以及身体、场景、源策略和配置哈希。加载时验证清单、
压缩档案和每一行哈希；NaN、篡改或混合合同均 fail-closed。

MJWarp 守门环境原来会在失败后立即隔离并恢复默认站姿。若在 `step()` 之后采样，
失败记忆会错误地记录为“站着”。现在环境在隔离前保存最后有限物理状态，采集器只读
该状态。这是把视频观感转成可训练因果数据的关键修复。

实际在 A6000 GPU0、64 个 MJWarp 世界运行冻结守门策略：

- 64 个剧集，11 次扑救，扑救率 17.19%；其中手部扑救 8 次，12.50%；
- 11 次扑救中 9 次随后失败，扑救后失败率 81.82%；
- 共得到 50 个真实快照：31 个高动量、11 个支撑/跪姿、7 个站立、1 个俯卧；
- 事件包括 11 个触球、16 个扑救后飞行、12 个恢复入口、9 个失败终态和 2 个剧集终态；
- 没有非有限状态被写入语料。

### 2. Memory：内容绑定的失败重放

新增确定性 replay sampler。失败样本、高动量样本和稀有姿态有更高采样权重，同时
禁止混合不同身体、场景、源策略和配置。这样持续学习会反复攻克失败簇，而不是被大量
简单站立样本稀释，也不会把不兼容实验混成一个数据集。

### 3. Growth：两阶段恢复课程

新增物理沉降器，仅使用关节阻尼和接触动力学，不修改根速度、不 teleport。A6000
GPU3 上把 9 个高动量终态转成 9 个低动量训练入口，无 NaN；结果为 8 个俯卧、1 个
左侧卧。它既验证了冲量吸收子问题，也暴露出当前真实数据仍缺右侧卧和仰卧覆盖。

课程被显式分为：

- A：从真实高动量快照训练有界 residual actor-critic，目标是安全卸力；
- B：从真实沉降快照及其小型物理扰动训练左/右侧卧、俯卧、仰卧 tracking-RL 专家；
- C：参考 OpenTrack，以 DAgger 将路由专家蒸馏为单一本体感恢复策略；
- D：回到完整“射门—扑救—落地—起身—站稳—继续守门”剧集进行成对晋升。

当前审计结果为 A 可开始，B 因缺真实右侧卧/仰卧覆盖而未就绪，C 的工具链可用但要
等待物理专家，D 未就绪。关键缺口被机器可读地标记为
`TRUE_SETTLED_LEFT_RIGHT_PRONE_SUPINE_POST_SAVE_COVERAGE`。

### 4. 修复 MotionDecode 教师片段

旧适配器对 recovery 文件固定截取最后 240 帧。多数 MotionDecode 文件的结构是
“站立—跌倒—起身—站立”，因此旧窗口常常只包含已经站好的尾部，实际上没有教起身。

新适配器以最低骨盆帧为起点，找到第一次连续站立帧，再保留 60 帧终端保持；同时从
Lie、Prone 和 Fall 三个族选择左侧、右侧、俯卧、仰卧教师。实际 v2 库已经选出四种
姿态教师，库哈希为：
`sha256:04b0edb9fa45fc64b669e1f426567ca6e39b4a23179570f4a1f9ff380f421244`。

对当前 9 个真实沉降状态，9/9 都能匹配同姿态 MotionDecode 片段。不过该结果只表示
教师路由正确，权限明确为 `TRAIN_ONLY_KINEMATIC_REFERENCE`，不是 9/9 物理起身成功。

## 四、ROSClaw 的完整学习链路

```text
Practice：连续物理剧集
  射门 → 扑救 → 落地 → 恢复 → ready
                 │
                 └─ 保存隔离前真实状态、接触、动作、上下文哈希
                              ↓
Memory：不可变失败语料 + 稀有/困难簇重放
                              ↓
Growth A：高动量冲量吸收 residual actor-critic
                              ↓
Growth B：动作教师 + 物理 tracking RL 多姿态起身专家
                              ↓
Growth C：多专家 DAgger → 单一本体感学生
                              ↓
完整链路成对考试：新候选 vs 冻结冠军
  同一射门种子、同一场景、无 auto-reset 计分
                              ↓
仅同时守住扑救率、伤害/稳定性和恢复时间才晋升
                              └─ 失败回流 Memory，进入下一轮成长
```

这解决 stability-plasticity dilemma 的方式不是“永远在线改当前 actor”，而是：冻结
冠军负责稳定性，训练候选负责可塑性；失败重放和旧技能回放防遗忘；候选先在隔离仿真
中成长，只有成对门控胜出才替换冠军。在线部分主要收集新分布、更新记忆和触发训练，
不让未经验证的权重直接获得运动权限。

## 五、后续实施门槛

下一轮应按以下顺序进行，不能跳到宣传视频：

1. 扩大真实扑救采集，按左右远角、高低球、二次射门和身体侧别定向采样，直到沉降后
   左侧、右侧、俯卧、仰卧每类至少 64 个独立状态；
2. 先训练 Stage A，要求所有困难簇都把角速度压入起身入口，同时不增加头部冲击、
   关节限位、能耗和扑救后滑移；
3. 对四类姿态分别做 MotionDecode/MOSAIC/LAFAN tracking RL，教师只提供参考相位，
   奖励必须包含双足站立、低线/角速度、接触安全和连续保持；
4. 用 DAgger 蒸馏成无姿态标签、只依赖本体感的统一恢复 actor，再接 S48 热交接；
5. 完整考试至少覆盖 1,000 个固定留出困难射门；主指标是扑救后 10 秒内恢复 ready 的
   条件成功率，而不是单独起身成功率；同时报告扑救率、手扑率、首次冲击、二次碰撞、
   恢复时间 P50/P95 和最终连续稳定时间；
6. 候选必须相对冻结冠军保持或提高扑救率，并在恢复率和安全指标上显著改善，才允许
   生成晋升视频。视频只做结果展示，不作为计分证据。

## 六、验证与证据

代码验证：轻量环境全套 355 passed、95 skipped；带 Torch/MJWarp 的 Isaac 环境
全套 438 passed、12 skipped；S49 与相关 MJWarp/MJLab 聚焦回归 31 passed、1 skipped；
Ruff 与 mypy 通过。GPU 环境的主要跳过项是未安装 ONNXRuntime，以及测试未自动发现
外部 G1 资产；本轮神经起身和真实快照物理考试已使用显式资产路径单独运行，因此不会
把这些 skip 记作物理能力通过。

主要原始证据：

- `/code/rosclaw/rosclaw_football/evidence/s49-full-chain-recovery-snapshots-v1/collection-report.json`
- `/code/rosclaw/rosclaw_football/evidence/s49-full-chain-recovery-snapshots-v1/goalkeeper-post-save-hard-corners.json`
- `/code/rosclaw/rosclaw_football/evidence/s49-full-chain-recovery-snapshots-v1/neural-getup-failure-terminal-exam.json`
- `/code/rosclaw/rosclaw_football/evidence/s49-full-chain-recovery-snapshots-v1/neural-getup-impact-absorption-exam.json`
- `/code/rosclaw/rosclaw_football/evidence/s49-settled-recovery-curriculum-v1/settling-report.json`
- `/code/rosclaw/rosclaw_football/evidence/s49-settled-recovery-curriculum-v1/source-audit.json`
- `/code/rosclaw/rosclaw_football/evidence/s49-settled-recovery-curriculum-v1/motiondecode-recovery-library-v2.json`
- `/code/rosclaw/rosclaw_football/evidence/s49-settled-recovery-curriculum-v1/teacher-routing.json`

全部实验为 `SIM_ONLY`，真实机器人命令为 0；没有硬件授权或现实部署声明。
