# RSI-M0 条件化触球选择器：发现集→留出集→视频（2026-09-24）

本轮在冻结 SONIC 小脑上增加 **有界、平滑的左腿触球目标残差**（左髋 −0.15 rad、左膝 +0.15 rad），再从物理课程中蒸馏一个仅决定接近方向的单阈值选择器。它不是端到端神经网络、不是在线 RL，也没有更新 SONIC 权重或取得任何硬件执行权；所有操作 `SIM_ONLY`。

## 训练与失败归因

- `scripts/rsi_sonic_contact_grid.py` 预声明 3 个球距 × 3 个横向球位 × 3 个横向接近速度，并各附一个冻结父策略对照，共 **36 次** 500 Hz MuJoCo 完整积分。外部证据：`/data/rosclaw_overflow/rsi-sonic-contact-grid-20260924-v1`，manifest `sha256:2791af9e4278014b312aea7e98009890375595fd800b3ceae3ace61a90e34a6f`。
- 粗略“整球入门”为 21/36；严格要求**首次机器人—球碰撞为脚**后仅 17/36。多个“进球”实际上是小腿先撞球，不能当作足球技能正例。9 个父策略直线球位中为 5/9 脚先入门；对每个训练球位事后从三种接近速度中挑选，9/9 都能找到入门解，但这是**训练集 oracle 上界**，不是学到的控制器成绩。
- 在训练集之外的开发试验中，候选在一个球位虽提高球速仍射偏；线性方向修正把成功从一球位搬到另一球位，2 个新球位仅 1/2，未超父策略。因此没有沿用线性规则。学习器以脚先入门为硬标签，额外惩罚不必要的横向运动，拟合出可解释的 `y=0.0875 m` 分界。来自格点与开发试验共 12 个物理成功标签，12/12 可被该阈值拟合，但同样不构成泛化证明。冻结模型 `/data/rosclaw_overflow/rsi-sonic-contact-selector-20260924-v1.json`，哈希 `sha256:106d1dd43c992d573c778a48e9cb07092d2ffa251c9158b30228a4debe92cd58`。
- 模型只输出 `-0.10` 或 `0.00 m/s` 横向接近指令，并且仅限 `SIM_ONLY`、球距 1.3–1.9 m、横向球位 0–0.2 m。底层 29 关节 SONIC、PD 力矩和物理时钟不由选择器改写。报告中的 `trained_actor=false` 是刻意准确的：这是物理数据蒸馏出的课程选择器，不是 learned motor actor。

## 冻结后留出集

在训练/开发之后，才声明四个未见过的 1.8 m 球距，横向球位 0.04、0.08、0.12、0.16 m。候选和父策略各做 4 次完整物理运行，共 8 次；同一球位的两者初始物理状态哈希一致。原始数据：`/data/rosclaw_overflow/rsi-sonic-contact-selector-holdout-20260924-v1`，manifest `sha256:ce45aec6a04a1ee988853a34dad2b110a80c691ed620557ee73052c507137047`。

| 新球位 y | 冻结父策略 | 候选选择器 | 候选球速峰值 | 候选最低骨盆 |
|---:|---|---|---:|---:|
| 0.04 m | 未进 | 脚先入门 | 3.018 m/s | 0.717 m |
| 0.08 m | 未进 | 脚先入门 | 2.743 m/s | 0.717 m |
| 0.12 m | 脚先入门 | 脚先入门 | 2.389 m/s | 0.697 m |
| 0.16 m | 脚先入门 | 脚先入门 | 2.365 m/s | 0.697 m |

`verify_sonic_contact_holdout.py` 对 8 次轨迹独立检查 hash、课程、父/子初态一致、50/500 Hz 状态吻合，并在重新建立的同一物理模型上从 500 Hz 位置重算首次碰撞部位，再复算整球入门、峰值球速和骨盆高度。结论 **候选 4/4，父策略 2/4**，verification `sha256:3b947274731cbffb0d3eed282a6d4d6f9aff98ac51f0c9940a328549964d80b1`。另有测试覆盖原始轨迹篡改、manifest 总数伪造和选择器哈希/越界拒绝。

这是一个真实的窄域改善，但**4 个新场景不够晋升**：只测试了同一球距、单球、静止球、同一 SONIC 版本/球门/地面；没有运动中的球、守门员、队友、逆风、不同初态或跨引擎复现。留出集结果不得外推成一般射门成功率。

### 第二组更远球距的反证

在首次 1.8 m 留出集之后，保持同一个选择器权重/阈值不变，预声明 1.9 m 球距的相同 4 个横向位置。新证据 `/data/rosclaw_overflow/rsi-sonic-contact-selector-holdout-20260924-x190-v1`；manifest `sha256:d75ad1e3cc8519b4d3ed186fdafbf0e6097d9f3c8873a08f906f912431706a0f`，独立审计 `sha256:dd702fe77dc85990b3a5288ea446e4f619862710271e461dcaa5ace0eb5d3867`。候选 **2/4**、父策略 **1/4**；两组留出集合计候选 **6/8**、父策略 **3/8**。1.9 m 的 y=0.12 和 0.16 m 课程中，候选首次接触分别是右小腿和左小腿；后者虽然球穿过球门，也必须判失败。球距变化造成的触球腿/相位切换是现有单阈值选择器无法处理的失稳点。此反证进一步排除了晋升和“高成功率射门”宣传结论，也说明需要显式观测步态相位与左右脚可达性，而不能只看球的横向位置。

## 可视化与后续

25 秒 1080p 四球位发展视频：`/data/rosclaw_overflow/rsi-sonic-contact-holdout-four-goals-20260924-1080p-v2.mp4`，同名 `.json` 保存 `sha256:855ccbfe21974da405e92e9f78eb9bbfbf3c63501d6d35ef343d44156c87e4ed` 视频哈希及物理审计关联。每段标注球位、父策略是否入门及候选出球速度。镜头和慢动作仅发生在物理积分后，画面像素不用于选优。现阶段视频可展示**可审计的四球位成长实验**，尚不适合作为“顶级球员”宣传片：步态仍偏慢且僵硬，射门多为低平球，只有一台 G1。

尝试将同一配置切换 SONIC v1.1：仍站立、脚先触球，但单题未进且球速仅 1.682 m/s；low-latency 对照进门、2.743 m/s。故不因另一模型名义上更新就盲目切换。接下来应在 MotionDecode/SONIC 身体先验上训练**连续运球/助跑→主动摆腿→踢后恢复**，并以跨球距、球速、目标高度、初态的 fresh holdout 测试自然度、球门精度、脚先接触率和失稳率；在此之前保持 `promotion_authorized=false`。

复核命令：

```bash
PYTHONPATH=src:/code/rosclaw/rosclaw_test/src /code/rosclaw/rosclaw_soccer_s2/.venv/bin/python \
  -m rosclaw_soccer.rsi.verify_sonic_contact_holdout \
  --root /data/rosclaw_overflow/rsi-sonic-contact-selector-holdout-20260924-v1 \
  --stadium-assets /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy
```
