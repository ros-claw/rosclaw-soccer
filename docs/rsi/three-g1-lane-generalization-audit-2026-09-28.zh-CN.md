# 三台 G1 新球位泛化审计（2026-09-28）

状态：`SIM_ONLY / REJECTED_DEVELOPMENT`。**没有新晋升策略，也没有新合格宣传片。**

## 为什么这轮不剪片

历史 S85 的四条球位曾全部通过传球→射门→手套扑救的 14 项物理门控，但它们是固定球位，不能代表相邻球位的泛化。按《ROSClaw Physical RSI 实施总纲》“Fresh Physical Evaluation 先于晋升”的原则，本轮事先声明四条不同于 S85 的横向球位，分别用 S85 门将 actor B 和同一训练群体的 actor A 测试。每个配置/球位跑两次 CPU MuJoCo 严格回放；没有用画面评分，也没有手动改球状态。

| 新球位 | actor B 的主要失败 | actor A 的主要失败 |
| --- | --- | --- |
| inside-right | 未形成有序射门→手套接触 | 同样未形成有序接触 |
| inside-center | 手套碰球后仍进门 | 仍进门；双手准备和向外拨球也未达标 |
| outside-left | 扑后速度超门、双手准备未达标 | 双手准备未达标 |
| wide-left | 向外拨球不足、仍进门 | 仍进门、双手准备未达标 |

两套 actor 均为 **0/4**；每套内部双次重放一致。四组球位是四个场景，不因重复或换 actor 而虚增为 16 个独立场景。旧的 S85 4/4 结果仍作为历史窄域样例保留；新结果说明它离能宣传的稳健球队能力还很远。完整失败状态、请求、轨迹和双次回放分别在外置目录：

- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s922-three-g1-holdout-showcase-v1/`
- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s922-three-g1-holdout-actor-a-v1/`

这些球位已进入 `CONSUMED_DEV`，今后不能再被称作 Fresh 或 Sealed。此轮也没有训练新权重，不能把“换了 actor”说成自进化。

## 物理资产边界

直接检查编译后的 MuJoCo `ball_geom`：球为半径 `0.115 m` 的球体，质量 `0.410 kg`，周长 `0.722566 m`。本项目的本地 IFAB 范围检查返回 `circumference_in_ifab_range=False`。球门为 `7.32 × 2.44 m` 不等于球也满足成人比赛规格；历史素材只可称“标准尺寸球门的仿真”，不能宣称全套比赛资产符合标准。

## 本轮工程加固

- 新增可复跑的四球位挑战入口 `scripts/run_three_g1_holdout_showcase.py`，只写仓库外 SIM 证据。
- 多球位视频的标题不再写死 `4/4`、`0.885 m`，而从实际证据计算数量和接触跨度；失败证据仍被媒体入口拒绝，不能借标题改动伪装成成功。
- 组合证据在物理执行前后比较实现文件、四个策略资产和请求文件哈希；运行时漂移则保留轨迹和失败报告，但组合晋升门 fail-closed。

## 下一步：回到真正的学习闭环

不继续用邻近球位试参数、筛视频。按总纲先冻结三 G1 的历史成功父代及这八组失败评估，明确 `TRAIN / CONSUMED_DEV / FRESH_HOLDOUT / SEALED` 分区；只开放一个 plastic boundary：球员的接球/首触适配器，身体基础、战术和安全壳冻结。先在单人滚动来球课程训练并考真实脚接触、触后控球、非脚碰球、全身关节/力矩/支撑稳定，再串联三 G1 传接射，最后考 4v4 连续比赛。晋升必须在未见物理课程上提高成功盆覆盖率且不丢旧能力；宣传片只能从通过的连续轨迹下游渲染。

本轮的 0/4 不是“继续加大渲染力度”的理由，而是现有门将/球队能力缺乏分布外稳定性的直接证据。
