# S10 因果分层守门员：开发实验报告

日期：2026-08-13
状态：`SIM_ONLY / REJECTED / NOT PROMOTED`

## 结论先行

本轮让守门员从“看见任意小动作就算反应”升级为：仅使用可见球历史估计速度、落点和六区域，再由神经 actor 决定时机/方向，合格 locomotion 小脑保持双腿平衡，受限单臂任务空间层尝试触球。它的确学会了新的低位右侧扑救，也让部分高球从完全够不到变成真实 MuJoCo 接触；但冻结 25 球考试仍只有 2/25，和父代持平，没有达到晋升门。因此本轮候选保留为反例和下一轮训练数据，不替换 Champion。

## 修复了什么

1. 观测契约升级为 V3：8 帧稳定本体坐标球历史、因果速度估计、弹道拦截时间/横向/高度、置信度和六区域 one-hot。actor 不读取射手 phase、未来轨迹或特权 critic。
2. 修复训练世界的伪横向数据：旧 fast world 的球历史横向位置不随时间变化，速度估计恒为零；现在使用真实对角弹道及启动期 padding。
3. 反应延迟改为“第一个朝目标方向的有效指令”，错误方向的小抖动不再冒充 0 ms 反应。
4. 加入分层动作所有权：神经 actor 负责意图和有界残差，locomotion 小脑独占双腿，单臂 DLS 层只写被选择的手臂。强制全身 checkpoint 直接混入曾导致骨盆跌至约 0.08 m，已明确否决。
5. 参考 `Humanoid-Goalkeeper` 的公开研究权重和动作数据，但采用许可隔离：不把 CC BY-NC-SA 权重放进仓库或 Champion，只生成内容寻址的外部 teacher bundle。
6. 手掌碰撞体改为不可见的手套形椭球；真实 MuJoCo 碰撞和计分共用同一 scene，不使用像素或距离伪造扑救。
7. 证据新增完整实现哈希，覆盖观测、actor、共享世界、场景碰撞和计分；实现变化会使旧证据/视频自动失配。

## 训练与测试

- 训练硬件：4 × NVIDIA RTX A6000，DDP 参数最大差 `0.0`。
- 有效训练样本：655,360。
- fast-world mean reward：`0.4044 → 0.6071`，best `0.6132`。
- fast-world save proxy：`0.5690`；recovery proxy：`0.8526`。
- 正式权重：`sha256:7db985b246270519b03438ddbb9279b46fb4b3992bec2efa4c27b4589e7d3006`。
- CPU MuJoCo 冻结考场：父代/候选各 25 球、各严格复跑一次，共 100 条轨迹；父代与候选均 `2/25` true saves、`2/25` contacts、`0` safety cost；严格复现与匹配场景均为 `true`。
- 扩展诊断网格（1.2 s、40 个边区）：父代 1 次 true save，候选 2 次，双方均无安全失败。候选在 `y=-0.40 m, z=0.60 m` 学会父代没有的真实救球，同时在另一侧仍存在接触回归。
- 晋升结论：`REJECTED`。延迟项通过，但 coverage、save-rate、恢复/二连扑、人类动作分、历史回归和 sealed holdout 未通过。

## 通俗解释

旧门将像是“球一来就扭一下身体”，系统把这个扭动误认为聪明反应；它其实不知道球会到哪里。现在它能凭连续几帧球的位置估算来球速度，并提前判断低右、高右或中路，再把任务分给脚下移动和手臂触球两个层级。它已经从“完全瞎扑”进到“个别方向会扑”，但球门宽度远大于当前慢速横移小脑的覆盖范围，高角球还需要真正的侧跨、跳扑及落地恢复动作族，而不是继续加大手臂力矩。

## 下一轮突破路径

1. 用本轮所有成功、漏球和安全边界样本建立对称左右 hard-negative curriculum，重点消除单侧偏置。
2. 训练可部署的侧跨/低扑/高扑/落地恢复技能族，并让 causal router 选择技能；不直接混入异构 simulator 的全身 checkpoint。
3. 把 `Humanoid-Goalkeeper` 六类真守门动作仅作为非商业 teacher，进行本地 G1 动力学重放与蒸馏，再由 CPU MuJoCo 验证。
4. 加入连续二连扑与扑后站稳指标；只有 fixed suite、holdout、动作自然度、恢复与安全全部过门，才允许晋升。

## 可核验证据

- 训练：`/code/rosclaw/rosclaw_football/evidence/s10-goalkeeper-causal-hierarchical-ppo-4gpu-dev-v2/`
- 冻结考试：`/code/rosclaw/rosclaw_football/evidence/s10-goalkeeper-causal-hierarchical-ppo-4gpu-dev-v2-cpu-exam-v2/`
- 24 秒 720p 对照视频：冻结考试目录中的 `development-v2.mp4`。视频明确标为 rejected development，画面不参与评分。
