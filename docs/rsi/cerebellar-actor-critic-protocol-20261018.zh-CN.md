# 小脑 actor-critic：持续更新器与下一物理学习阶段

## 当前状态，不虚报训练

实现了 v2 小脑的数值更新器和恢复学习目标，尚未用真实足球轨迹训练新
小脑。合成测试中的参数变化只证明优化器工作，不计入足球学习更新。
整个目标仍是连续自主球队比赛；本阶段只是底层稳定控制的一步。

独立复核已完成自适应 v1 三体实验的第一个 `original_none`：10 个 witness
文件哈希通过，33 个原始数组与旧轨迹 dtype、shape、bytes 完全一致；
原始状态重新标注及独立 CPU/Foundation qualification 一致。
receipt 为 `sha256:dafcae8ac90ce3b3fcf223b5df8882afee241faee4c94035171bd024d26720c0`。
首次脚触球 85 帧，前向 2.0932230597931056 m，侧向 -0.20936339739767457 m，
最低骨盆 0.6962456565911066 m，本次任务通过；没有新优化器更新。
随后 `neural_zero` 也已提交并独立复核：10 个 witness 文件哈希通过，
全部 33 个公共原始数组与 `original_none` 字节一致，新增 10 个小脑 capture
字段；3000 子步请求/实际残差和 mean 全为零，未采样；资格核对一致，任务通过。
receipt 为 `sha256:5b03aa67fa676833950297602e67c684b6a414269b8fa1b13877c12204a03fce`。
这证明本次零层中性，不是学习成果。

### 三体终态：接口闭环通过，尚未训练

实验于本地 09:22:18 退出，实际 systemd 状态 inactive、MainPID=0、Result=success。
随后独立检查全部输入/源码/witness 文件、SQLite 完整哈希链、三条完整 receipt、
原始测量重标注与原独立 CPU/Foundation 复核绑定。

| 模式 | 前向球路 m | 侧向球路 m | 最低骨盆 m | 任务通过 |
|---|---:|---:|---:|---|
| 原控制 None | 2.093223 | -0.209363 | 0.696246 | 是 |
| 零头小脑 | 2.093223 | -0.209363 | 0.696246 | 是 |
| 随机小脑 std=0.05 | 2.169608 | -0.197107 | 0.699877 | 是 |

三条首次脚触球都在 85 帧。随机小脑确实执行 3000 次潜变量采样并进入实际
关节力矩；公共 canonical qvel 与原控制最大差异 12.487949 rad/s，身体响应
不是仅记录了随机数。差异大不等于更自然，也不是 jerk 或学习提升。
三体合计新 native 9000 子步、独立 CPU 9000 子步、小脑 6000 次前向、
3000 次随机采样；新小脑优化器更新仍为 **0**。

同一个公开派生初态的三个控制，不是三个独立场景或普遍成功率。
相较之前固定旧目标序列的随机残差回放倒地，这支持在当前身体上持续
重算 Foundation/父代控制的必要性，但不能分离归因于 Foundation 或父代
motor 的具体反馈，也不能据一个例子声称解决自然性和稳定性。

complete：`sha256:a2f93f5ee9ffa5fef619d011dadecbbef21cacf96be250d13bf8cec6a7532a9b`。
外部终态读回 `rsi-adaptive-cerebellar-native-pilot-independent-readback-20261018.json`，
seal：`sha256:b30787c9a36b394c9cb6a5bbc5dab3b6ef9eeefa252f937fe5355837f45b8626`。
读回认证既有独立物理重演，没有再跑新物理，也没有在此重建共享大模型
payload 的逻辑报告 seal；这两项边界在报告中明确标为 false/0。

## 数值更新器

- actor：127→64→64→29；critic：127→64→1，使用同一有限期限状态。
- CPU float64，不改全局随机种子，不使用 GPU 或机器人运输层。
- 潜变量是原始 IID Gaussian，std=0.05，先采样再 tanh、限矩。
  PPO 概率不作用于硬限幅后的力矩。
- 输入必须是 2..8 条完整的 3000 子步 episode，包含全部成功和失败。
  验证旧 actor 身份、概率、顺序、时钟、终止状态、每轨迹完整 MC return。
- gamma=1，期限结束不 bootstrap；return scale=10，优势不做样本均值中心化，
  避免相同失败返回在零 critic 时被消成零信号。这不保证估计低方差。
- 每批最多 8 次全批更新，默认 4 次，Adam lr=3e-5，PPO clip=0.2。
- JSON 检查点保存 actor、critic、Adam 一阶/二阶矩和真实步数。
  重启后必须延续，不能用有学习步数的 actor 重新初始化成空 Adam。
- 固定初始 anchor actor 与特征字节哈希跨轮不变；保留 KL 均值/逐状态约束。
  旧状态 KL 有界不是旧物理能力的保证，仍需实际 retention 对照。
- 超出 rollout/anchor KL 上限则整步回滚，包括 Adam 状态，不暗中降率重试。
  receipt 区分尝试次数、接受次数、非零 actor 梯度次数。
  用过的数值批次摘要落盘，拒绝在下一轮悄悄再次消费同批数据。
- 优化器不认证物理来源；调用方须单独绑定 witness、独立重演与输入文件。

验证：新增优化器 14 项及恢复 return 9 项合成测试通过，相关 16 组回归
合计 463 passed；改动文件 Ruff/format/compileall 通过，两个新增源文件定向
mypy 通过。曾有一次回归命令漏设冻结 Core PYTHONPATH，产生 collection error；
用正确环境重跑后上述 463 项通过，不将该环境错误算成通过或跳过。
这些不是全仓测试或真实足球训练结果。

## 恢复学习目标，与正式标签分开

### 真实数据到更新器的入口

新增 `cerebellar_training_batch.build_batch`：读取实际冻结声明，要求该 actor
全部 collection jobs 按原顺序输入，不能只传调用方自选子集；检查显式
SIM 研究训练范围，并拒绝 v1 数据冒充 v2、旧模型/世界/随机种子错配。
逐文件验证 receipt 和全部 witness，重建共享 payload 的原始完整物理报告
seal，核对其 commitment/原始 trace/独立 CPU 与 Foundation review 绑定，
核对 15 次 planner reference 重建与 300×640 encoder 特征来源。
再独立重算 3000 子步本体输入、实际潜变量/概率和学习 return。
合法 unsafe、无触球、右删失仍保留为完整失败轨迹，不能触发成功筛选。

新增 13 项合成 packet/文件管线测试；文件管线测试明确 mock 物理报告读者，
不是新物理证据。相关回归 476 项通过，新增源文件定向 mypy、Ruff 通过。
生产入口尚待新 v2 真实采集运行；目前仍没有新增足球训练更新。

正式 300 帧任务、完整触球后 60 帧球路、足部接触与骨盆安全门均不变。
新学习 return = 正式 training_reward + 5×task_pass − 有界恢复代价。

恢复代价在首次脚触球帧之后的 60 个完整控制帧，用 600 个真实积分前状态计算：

`min(2, 0.25×mean(||root_angular_velocity||²)
        + 0.01×mean(joint_velocity²)
        + 0.01×mean((actually_applied_residual/cap)²))`

不存在完整窗口时，诊断是 null，而不是“完美稳定”；代价项不伪造测量，
使用正式无触球/右删失/安全失败标签。摔倒仍是正式失败，不能用辅助奖励
洗成通过。辅助 return 是整段 MC 目标，尚不是逐步 GAE，也不声称自然性已解决。

## 下一阶段有限实验设计（待最终冻结、preflight 和执行，不是已启动）

准备进度：已在冻结源 `1bfefdc` 上，从八条旧成功控制的全部 300 帧构建
2400×127 维锚点，逐一检查原有 witness、标签及 qualification。
三个 seed 的 actor/critic/Adam JSON 初始检查点已生成，更新步数均为 0；
没有新增物理执行，没有把旧数据当新模型的 on-policy rollout。
外部证据目录 `rsi-cerebellar-ac-initialization-20261018`，complete seal 为
`sha256:85558a7398a9e5071612877515aad687b9f1ac3552248f0d86578892f58dba76`。
八条源轨迹的 compiled world 文件哈希均为原定
`sha256:246da475bbad8d3ced351f6ceb18e18fc3d4c2de7236cdd8c427239cecb4ff83`。

1. 等当前三体实验全部结束并复核完整性。合法随机失败保留为数据，不要求
   随机小脑先成功才允许学习；基础设施错误必须修复而非无限重试。
2. 固定三个初始 seed：2026101801、2026101802、2026101803。
   父代 Foundation、12D motor、世界参数与 H=300 不变。
3. 新学习使用旧 8 个公开 planning contexts 的偶数索引 0/2/4/6，覆盖四类
   初始距离/速度方向；每 seed 四条独立采样，共 12 条真实新 V2 episode。
   不把 v1 轨迹重命名成 v2 on-policy 数据。
4. 父代高层 option 固定为旧研究预先选出的各 context 最佳 option，不再搜索：
   0..7 分别是 `[3,7,0,0,2,0,1,2]`。这是既有先验，不是本轮学习成绩。
5. 学习前先做 8 个零头控制：四个奇数 context 的原先选定父代、四个历史
   A 成功案例（groups 4/8/32/56），核对物理回放和原有公共数组字节中性。
6. 每 seed 使用自己的四条完整采样轨迹进行一次默认 4-step 更新，最多
   12 个接受的联合 Adam 步；不得只保留好的轨迹或挑最好 seed。
7. 全部训练模型冻结后，三个 seed 各自评估四个奇数 context，并各评估四个
   历史成功案例，共 24 条新确定性 V2 episode。与前述零头控制配对比较。
8. 单阶段上限 44 个新 native body（8 控制+12 采集+24 评估），各自还需
   独立 CPU MuJoCo 重演和 300 次 Foundation 复算。两个 full-model worker
   同时运行的上限不变；现在已有两个在跑，不能擅自再开第三个。

奇偶 contexts 是同四个公开父场景的派生样本，不能叫 ancestry-disjoint 或
private Fresh 考试。分 seed 报告通过、失败、安全回归、恢复代价及其原始
分量；任何“学好了”的说法都必须有新物理对照，不能从 PPO loss 下降推出。

全部通过也不等于球队宣传目标完成。后面还要将真正可用的小脑接到各角色
身体闭环，再验证接球、带球、传射、扑救及连续 4v4 比赛。
