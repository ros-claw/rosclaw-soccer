# MARLadona 参考审计：借鉴团队表示，不迁移框架

2026-09-21，固定官方提交 `8a24b2355518de0f39a1825d0961c3285a4c5a17`。
[仓库及许可证](https://github.com/leggedrobotics/marladona-isaac-lab/tree/8a24b2355518de0f39a1825d0961c3285a4c5a17)，
Apache-2.0。已获取到外置参考目录，未安装其环境、执行训练或导入其历史权重。
本次不迁移到 Isaac Gym，也不更换 ROSClaw 主框架。

## 已读实现与可借鉴内容

- `source/rsl_marl/rsl_marl/modules/neighbor_net.py`：分别编码队友和对手，
  对有效实体做池化。说明可用集合表示支持不同队伍规模，不需要把每个球员
  的输入槽硬编码成不同网络；这不等于取消每个球员自己的角色和状态。
- `modules/policy_replay_manager.py`：历史策略快照池与按环境分配的对手版本。
  ROSClaw 应继续用自己的版本哈希、冻结状态、League 和证据约束。
- `runners/on_policy_runner.py`：训练 actor 与独立评估 actor 的索引分开，
  actor 和 critic 观测也明确分开；不能把 critic 特权信息漏给部署 actor。
- `isaaclab_marl/tasks/soccer/mdp/observations.py`：统一攻防坐标系，分开
  自身状态、球、队友和对手。借鉴语义，不能直接复制不同坐标尺度的张量。

## 不能原样照搬的细节

这些是固定提交的源码观察，不是完整运行审计或上游漏洞定论：

- 缺席实体用 NaN 标记。当前 ROSClaw 的非有限输入 fail-closed 约束需要显式
  validity mask，不能把 NaN 当普通数值穿过运行时边界。
- NeighborNet 对全缺席队友的池化无穷值有替换，而对手侧未见同样替换；
  我们需测试零邻居、全 mask、实体顺序置换和不同人数，不假定上游已覆盖。
- replay buffer 在类层声明可变字典；我们的池应实例隔离，禁止不同实验串池。
- 历史权重加载没有本项目所需的内容哈希与安全张量合同；不运行未验证 pickle。
- runner 使用配置字符串 `eval` 选类；本项目应使用显式注册表，不扩张执行面。

## 对当前实施的约束

先做非 29DoF、5–10 Hz 的 2v1 决策世界，再用 MAPPO + 历史对手池。
Human Tactical Prior 只作初始化/辅助监督，不将人类位移直接变成 G1 控制。
技能成功率、时长、readiness 来自物理证据；缺失项仍是未知，不用手填数字
制造好看的传球收益。当前还没有完成这个团队闭环或证明 MAPPO 带来收益。
