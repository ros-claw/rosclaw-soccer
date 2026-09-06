# S201：连续链最早失败归因与传球残差学习

日期：2026-09-06

状态：`PASS_ROLE_LOCAL_FAILURE_CREDIT` / `SIM_ONLY`

物理权威：S200 CPU MuJoCo 双次严格回放；视频像素不参与评分

## 结论先行

本轮没有把“踢到球”误报成“传球成功”。六台 G1 同场 rollout 中，playmaker
确实由右脚触球，峰值球速为 `2.332 m/s`，全队安全门也通过；但球到自治
finisher 目标的最小三维误差仍为 `1.284 m`。新的连续链评分器因此在 `PASS`
阶段停止，把最早因果失败只归给 `red.playmaker`，冻结其余五个 ROSClaw cell，
不再让坏传球污染前锋射门或门将扑救的学习数据。

本轮还第一次把“战术交付点”和“全身击球策略的物理瞄准点”分开建模。从严格
回放轨迹拟合得到初始瞄准残差 `(+1.239, +0.317, -0.115) m`。这是数据驱动的
seed actor，不是手写补丁；但它只有一个独立物理上下文，因而被硬性标记为
`NOT DEPLOYABLE`。重复同一轨迹或只改结果文件名不能伪造样本覆盖率。

## 为什么旧方案接不上

参数辨识显示问题不是一个脚尖角度：

| 脚偏航 | 触球后峰值球速 | 目标误差 | 安全 |
|---:|---:|---:|:---:|
| `-0.04 rad` | `0.879 m/s` | `1.412 m` | 否 |
| `0.00 rad` | `0.645 m/s` | `1.474 m` | 否 |
| `0.02 rad` | `0.508 m/s` | `1.650 m` | 否 |
| `0.04 rad`（父代） | `2.332 m/s` | `1.284 m` | 是 |
| `0.06 rad` | `0.206 m/s` | `1.895 m` | 是，但过慢 |

减小偏航会牺牲支撑稳定性，增大偏航又会失去有效冲量。另一个物理试验保持父代
击球动作，并把 finisher 放进预估通道：击球仍安全、球速不变，但 finisher 的旧
`RUN_IN_BEHIND` 策略持续把身体带离来球通道，未发生接球者触球。也就是说，真正
瓶颈是“传球瞄准模型、接球准备位和一脚射门相位”没有共享同一个到达预测。

## 本轮通用架构

### 同球连续 option 链

`continuous_option_chain.py` 定义 `PASS → RECEIVE → SHOOT → SAVE` 的严格前缀
状态机。每一阶段绑定角色、agent cell、Champion/Parent policy、decision、option
request、输入/输出球状态哈希以及起止时间。后继阶段的输入哈希必须等于前驱输出
哈希；换球、倒序、时间重叠、非确定性回放或赛后改球状态都会 fail closed。

`RECEIVE` 明确支持“一脚射门”的无额外触球准备阶段：准备位与相位需要通过，
真正的首次触球归属紧随其后的 `SHOOT`，不会为了填满四段而虚构一次接触。

### 最早因果失败与单角色 plasticity

评分器只选择最早失败：传球不准时，finisher 和 goalkeeper 的后续结果全部停止
记功/记过。只有可复现、同球、角色本地的失败才能申请 `PlasticityLease`；本轮租约
覆盖全部六个 cell，但仅 `red.playmaker=PLASTIC`，其余五个均为 `FROZEN`。
球谱系断裂或严格回放不一致属于系统证据故障，不给任何角色开学习权。

### 数据驱动瞄准残差 actor

`pass_aim_calibration.py` 从安全、触球、严格回放的物理样本学习
`physical_aim - observed_delivery`。小批在线数据用中位数抑制碰撞或滚动尾样本的
异常值。actor 绑定物理策略 hash、上下文 hash 和轨迹 digest；部署门要求至少四个
互不相同的上下文及四条互不相同的轨迹，本轮单样本只用于生成下一轮探索提案。

## 证据与视频

冻结目录：

```text
$ROSCLAW_SOCCER_EVIDENCE/athlete-foundation-v1/
  s201-independent-chain-credit-v4/
```

关键文件：

- `chain-credit-exam.json`：同球链请求、PASS 事件、最早失败、六 cell 租约与边界。
- `playmaker-pass-aim-residual-seed.json`：单上下文 residual seed actor。
- `s201-causal-failure-credit-final-1080p.mp4`：物理失败与学习路由诊断片。
- 同名 `.json`：视频、S200 原始视频和 S201 报告的完整哈希绑定。

报告 hash（实现最终冻结后重新生成）：见 `chain-credit-exam.json`。视频只是下游
可视化，`promotion_eligible=false`，不参与分数或 actor 选择。

## 诚实边界

本轮通过的是“可信失败进入正确角色的学习队列”，不是连续传—接—射—扑已经成功。
没有候选更新被应用，因此 Parent retention 状态是 `NOT_RUN_NO_CANDIDATE`；seed actor
也没有部署权。S201 解决的是此前最危险的信用污染问题，并把失败转成可训练数据，
但尚未证明新的 pass actor 在未见过的接球点上泛化。

## 下一阶段硬门槛

S202 已发现此前已有的 `dynamic_lead_pass` 是更合适的角色专用后端：它有 8 个不同
上下文、8 条不同轨迹、封存 holdout、严格回放和父策略对照。下一阶段不再用单样本
residual 盲目外推，而是先把成熟传球后端接入角色路由，再把同一颗球交给 finisher，
由连续链重新选择下一处最早失败。结果见
`s202-role-qualified-backend-and-causal-handoff.zh-CN.md`。
