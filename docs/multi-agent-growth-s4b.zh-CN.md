# S4b：三角色智能体共同成长基座

日期：2026-08-12

## 为什么旧角色在修正球物理后变弱

S4a 证明旧三机传球段几乎是纯滑移。把足球恢复为真实滚动以后，我用同一个共享世界、
同一组旧 passer / shooter / goalkeeper 控制器重新运行，得到：

| 指标 | 旧冻结证据 | 真实滚动基线 |
| --- | ---: | ---: |
| 传球接收误差 | 0.036 m | 0.323 m |
| 射门目标误差 | 0.0018 m | 2.919 m |
| 门将横移 | 1.206 m | 0.244 m |

这不是“修坏了足球”，而是旧策略学会了错误世界模型。球由滑动变成滚动后，到达时间、
触球点和飞行方向都发生变化，三个控制器必须在新物理分布下重新适配。

第一轮安全时序搜索把 receiver 启动从 2.50 s 移到 2.10 s：接球误差降至 0.073 m，
射门误差降至 0.289 m，且没有跌倒、关节越界或扭矩越界。2.19–2.21 s 的样本虽然把
目标误差进一步降到 0.025–0.073 m，却发生关节越界，所以明确拒绝，不能拿漂亮结果
掩盖安全失败。

第二轮没有放宽越界判据，而是增强 shooter 的速度感知关节边界投影：margin 0.10 rad、
预测时域 0.20 s、`kp/kd=180/18`。在 2.19 s 相位上得到接球误差 0.0235 m、射门误差
0.0551 m、最低 shooter pelvis 0.675 m，且 passer / shooter / goalkeeper 均无关节越界、
无跌倒。它证明“更准”和“更安全”可以同时改善；但目前仍是迁移 runner 上的 bounded
diagnostic，不是新三角色晋级 evidence。

门将仍是当前瓶颈。把反应延迟从 0.12 s 降至 0.08 s、横移增益升至 2.5、速度上限升至
0.40 m/s 后，安全横移仅由 0.104 m 提高到 0.159 m，仍未触球。原因不是简单增益不足，
而是只等射门触球后才反应，面对约 9 m/s 的近距离球，剩余时间不够。下一代 keeper
必须学习射手姿态/支撑脚/来球相位的预判，并具备扑救动作；不能继续只调横移 PID。

## 每个角色如何成为智能体

新增 `role_learning.py`，三种角色都有独立的：

- `agent_id`、policy version、artifact hash、parent lineage 和 generation；
- observation / action contract；
- executed action trace；
- individual reward、side reward、stability score 和 safety cost；
- matched policy ablation 产生的反事实 evidence。

反事实还必须绑定同一个 seed、scenario hash、environment hash，以及被替换的 parent
artifact；换世界、换随机种子或用不相干的旧回放都无法冒充角色贡献。

训练可以采用 centralized training / decentralized execution：critic 可看共享球场状态，
执行时每个 actor 只获得自己的角色观察和动作权限。

本轮还新增 `joint_policy_search.py`，不再只靠手改补丁参数。passer、shooter 和
goalkeeper 各自拥有独立、带边界的策略向量；同一共享世界 seed 下执行 `+epsilon` 与
`-epsilon` 镜像实验，从真实成功/失败分数估计角色局部更新方向。任何跌倒、越界、非
严格回放或滚动失败的 probe 都不能给策略加权；若某角色剩余安全样本不足或没有学习
信号，则整组三角色 proposal 失败。该搜索只产生 candidate，不能绕过下方 holdout 门。

## 为什么不用一个“全队进球奖励”

只给全队一个进球奖励会产生 free-rider：传球者可以传烂球，只要射手偶然救回来也拿满
分；门将和攻击方又是对手，不能共享同方向目标。

S4b 使用 side-aware difference reward：

```text
D_i = 当前 side reward - 替换角色 i 为其 parent policy 后的 side reward
```

反事实必须来自同 seed、同环境、真实 MuJoCo 策略消融 evidence，不能由 critic 预测代替。
passer 和 shooter 属于 attack side；goalkeeper 属于 defense side，因此守门员可以通过
扑救能力成长，而不需要帮助攻击方进球。

## 联合晋级门

`evaluate_joint_growth()` 在相同 seed 的 parent / candidate 套件上检查：

1. passer、shooter、goalkeeper 三个 policy artifact 都已改变；
2. 每个角色 individual reward 提升；
3. 每个角色 difference reward 提升；
4. difference reward 的低尾 CVaR 不回退；
5. worst-case stability 不超过容许回退；
6. safety cost 必须为零；
7. strict replay、真实滚动和物理事件顺序全部通过；
8. 视频像素不参与评分，始终 `SIM_ONLY / CPU_MUJOCO`。

任何一个角色不成长、搭便车、跌倒、越界或证据缺失，整代都不能晋级。

发现集通过仍不够：`evaluate_joint_growth_round()` 要求同一组三角色候选在随机种子完全
隔离的 holdout 再通过一次，才原子化返回三份 promoted policy。训练集成功但 holdout
退化时，三个角色一起保留 parent，不产生“半支新队伍”。

## 本轮同时完成的基础修复

- 球门网升级为 target-independent 的三轴形变口袋：以首次实际触网点为锚，不以射门
  目标为锚，避免球进网后像掉进水里或被目标点“吸走”；
- 单 G1 free-kick 执行器已经复用该 stateful net；
- Soccer Growth adapter 增加 `soccer.shooting` 和 `soccer.goalkeeping`；
- Soccer SimForge task provider 增加 `soccer.three_role_league`，只允许修改三个角色策略
  路径，至少 8 个 seed 并要求 holdout。

## 当前诚实边界

S4a 视频是本阶段滚动修复的可视化证据：

`/code/rosclaw/rosclaw_football/evidence/s4a-rolling-video-v1/s4a-football-rolling-before-after-1080p.mp4`

S4b 目前完成的是角色学习合同、真实滚动下的训练诊断和两轮安全搜索，不声称三角色已
全部训练成功。旧三角色视频已被 S4a validator 判为滑移失败，不能复用。下一份三角色
视频必须来自迁移到 Soccer 后的新共享世界 runner，并绑定全部三角色 policy artifact、
反事实 evidence、滚动真实性和联合晋级报告。
