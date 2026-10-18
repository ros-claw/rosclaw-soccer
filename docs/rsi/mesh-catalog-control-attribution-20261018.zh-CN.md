# 成功控制实际做了什么：不能从option名称推断动作权限

对已完整提交的option4/6、raw8/56四条轨迹做只读检查，没有新增物理、
优化或标签。下表统计原始 `taskspace_joint_increment_rad` 中绝对值
大于1e−9rad的控制帧（包括触球后既有release尾段）：

| 收据job | 非零帧数 | 首/末控制帧 | 最大单关节增量 | 首足触帧 |
|---|---:|---|---:|---:|
| option4 raw8 | 28 | 69–96 | 0.302781rad | 77 |
| option4 raw56 | 29 | 59–96 | 0.291513rad | 77 |
| option6 raw8 | 27 | 66–92 | 0.230912rad | 73 |
| option6 raw56 | 30 | 55–93 | 0.270224rad | 74 |

四条都实际使用了脚部任务空间增量，不能把option6的`legacy` swing标签
误解成“只有B导航、完全没有C脚部控制”。当bounded_control_profile存在，
native采用continuous_swing_joint_delta；swing_mode还影响资格门控。

具体语义：

- legacy：需要旧learned memory gate且初始vx<0；
- memory_gate_both_directions_diagnostic：保留memory gate，去掉vx限制；
- bounded_contact_diagnostic：明确绕过memory与vx两个资格门；
- 三者都受foundation_only排除与下游关节/力矩守卫约束。

旧模块顶层注释将两种diagnostic模式混为一谈，现仅修正文档，并在已有
20组输入组合回归中同时断言bounded模式。函数实现未改变；运行中源
7b112e83与8d748da6不编辑、不重新绑定新hash。

option4/6还同时改变多项接近与脚部参数，所以这四条结果也不能因果
归因于某一个参数、gate或IK。它们是“既有组合控制能救回”的证据；
若要回答A/B/C各自贡献，仍需保持其它项相同的独立物理消融。
目前的18体扩展只验证所选组合的跨状态/保持性，不冒充这项消融。

本轮三seed学习失败与catalog成功也不能合并成“RL带来了改善”：
前者真实6次接受更新且无净收益，后者新更新为零、旧catalog复用。
