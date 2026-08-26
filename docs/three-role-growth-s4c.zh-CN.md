# S4c：真实滚动三智能体闭环与门将本体预判

日期：2026-08-12

## 本轮结果

本轮把历史实验目录中的 passer / shooter / goalkeeper 同场 MuJoCo runner 正式迁入
`rosclaw_soccer.skills.team`，并打通 ROSClaw CLI：

```bash
python -m rosclaw.entrypoint soccer academy train-three-role ...
```

一个命令现在会完成资产资格检查、候选 rollout、逐数组严格重放、反事实 reactive keeper
基线、真实滚动审计，以及 hash 绑定证据输出。它全程为 `SIM_ONLY / CPU_MUJOCO`，不打开
ROS、DDS、厂商 SDK 或硬件。

在修正后的真实滚动足球上，本轮 retained attack candidate 得到：

| 指标 | 结果 |
| --- | ---: |
| 传球距离 | 2.847 m |
| 接球误差 | 0.0235 m |
| 射门距离 | 6.663 m |
| 射门球速 | 9.074 m/s |
| 目标误差 | 0.0551 m |
| 滚动滑移比中位数 | 0.0762，通过 |
| passer / shooter 最低骨盆 | 0.681 / 0.675 m |
| 跌倒、关节越界、扭矩越界 | 全部没有 |

目标误差低于 0.10 m，且 shooter 的安全关节预测投影使用 0.10 rad margin、0.20 s
horizon、`kp/kd=180/18`，没有通过放宽安全门换取精度。

## 门将为何必须在触球前反应

旧门将只在球离开射手脚以后看球速。9 m/s 的近距离射门留给横移策略的时间太短；把
PID 增益拉满只能把安全横移提高到约 0.14 m，仍无法接近扑救点。

新 keeper actor 在 shooter policy frame、射门脚与球距离进入冻结窗口后开始预判。输入
来自共享世界中的本体和球状态，不读取未来轨迹，也不用渲染像素。相同射门轨迹的 A/B：

| keeper | 横移 | 最低骨盆 | 关节越界 | 扑球 |
| --- | ---: | ---: | --- | --- |
| 仅触球后反应 | 0.136 m | 0.734 m | 否 | 否 |
| 本体阶段预判 | 0.251 m | 0.751 m | 否 | 否 |

横移改善 84.7%，而攻击方轨迹、落点和安全结果逐位一致。说明提升来自门将更早获得因果
时间窗口，不是改球、改射门或录像作弊。

## 为什么仍然拒绝晋级

预判 keeper 尚未实际接触足球。因此开发 evidence 明确保存：

- `passed: false`；
- `promotion_status: REJECTED_DEVELOPMENT`；
- `goalkeeper_ball_contact_achieved: false`；
- `candidate_promoted: false`。

视频只能在显式 `--allow-rejected-candidate` 下渲染，画面底部永久标注
`REJECTED DEVELOPMENT CANDIDATE · NOT PROMOTED`。它可以用来判断动作效果，但不能被
当作三角色晋级证明。

## 自进化闭环的工程变化

- 三角色 runner、recovery controller、关节安全投影和 keeper actor 归属 Soccer，不再依赖
  历史 Core 足球脚本；
- evidence 绑定当前 runner、证据生成器、请求、轨迹内容和三个 role policy hash；
- goalkeeper 的联合搜索空间由六维 reactive PID 扩展为九维，新增预判起始帧、目标融合率
  和预判速度比例；
- 严格 validator 继续执行真实滚动、安全、精度和物理事件顺序门；development 例外只豁免
  旧的 0.75 m 横移门，并强制“有预判、未触球、明确拒绝”的一致语义；
- 媒体完全位于证据下游，像素不参与训练和晋级评分。

## 证据与视频

正式 CLI 证据：

`/code/rosclaw/rosclaw_football/evidence/s4c-three-role-anticipation-v4/`

本轮 37.73 秒、1920×1080、30 fps H.264 视频：

`/code/rosclaw/rosclaw_football/evidence/s4c-three-role-anticipation-video-v1/s4c-three-g1-proprioceptive-keeper-development-1080p.mp4`

视频 SHA-256：

`b960622853b691089df2834c14286d13b879ffbdf60bbc14d366a40d048ef2f2`

## 下一阶段

下一阶段不再继续增加 keeper 横移 PID，而是实现 role-owned goalkeeper skill：基于射手
支撑脚、摆腿相位、来球相位和目标先验输出离散动作（站位、侧步、跨步、侧扑、手臂拦截），
以实际 `ball_contact/save` 作为个体奖励。训练使用 matched attack seeds 和 difference
reward，只有在隔离 holdout 中提高扑救率、保持零安全成本且不牺牲其他角色稳定性时，三份
policy 才原子晋级。
