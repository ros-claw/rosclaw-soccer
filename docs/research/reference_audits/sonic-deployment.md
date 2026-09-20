# SONIC checkpoint 与 deployment 配置不能混为一谈

2026-09-20 核对官方 README，远端 HEAD 为
`7f151314d4d1606544bf249d2a7a1cb754c64582`。
本地参考 clone 仍固定为 `32c8260e54118b1f92b1fdeb9395d70d828e51a5`；
没有在运行中的 M0 pilot 内更新源码、checkpoint 或增益。

[官方模型和部署说明](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/7f151314d4d1606544bf249d2a7a1cb754c64582/README.md)
明确区分 low-latency 和 v1.1。80 ms 与 200 ms 是参考 lookahead，
不是总闭环实测延迟。

2026-08-31 更新中的 per-motor Kp/Kd 示例，是 v1.1 左右踝 pitch
硬件索引 4、10 的可选 1.5 倍配置。官方明确它不改变其他 checkpoint。
不能把这一配置默认套给本轮冻结的 low-latency，再归因于模型本身。

当前 Soccer 的通用 `joint_gain_scales` 仅接受 0.5–1.0，运行路径没有开启
上述 1.5 倍选项。若以后研究这一配置，需要新的显式契约、关节名与索引校验、
限矩测试和成对物理验收，不能绕过现有边界。

本轮 A2/A3 结论只适用于记录的 checkpoint、planner、增益与冷启动配置；
没有覆盖全部 SONIC 能力，也没有否定官方 v1.1 的稳定性改进。

## 2026-09-21 身体资产迁移审计

官方固定提交 XML 引用的 36 个 STL 已按 LFS 声明的大小和 SHA-256 下载到外部
审计目录，未改原参考 checkout。编译后官方资产身体质量约 35.112 kg，足球资产
约 33.341 kg；腰部质量/惯量、部分安装位置、关节 armature/frictionloss 以及
脚部碰撞几何存在差异。官方脚底使用离散小球，足球资产使用胶囊组合。
这些是资产对比，不代表官方训练环境或完整原生运行栈。

另作四次三秒诊断：两种资产各 primary/replay，冻结 low-latency 控制器、
相同初态关节/根姿态、相同生成参考，匹配地面材料和主要求解选项，不评价球技。
两边分别精确重放，参考哈希一致。官方资产最低骨盆高度约 0.124 m（跌倒），
足球资产约 0.684 m。右踝 pitch PD 跟踪 RMS 分别约 0.536 / 0.544 rad。
**直接替换官方 XML 并没有解决问题**；这是多项身体差异捆绑的迁移测试，
不能归因于某一个参数，也不是官方 SONIC 原生质量结论。不修改冻结 M0 物理。

外置证据：`sonic-body-transfer-audit-v1.json`、
`sonic-official-audit-assets/manifest.json`、`sonic-body-transfer-probe-v1/complete.json`。

## 静止参考与零命令复查

为分离“跑动参考困难”和资产迁移因素，另外执行 8 次三秒 CPU MuJoCo 诊断：
两种资产 × 固定初始姿态/零速度 planner × primary/replay。逐个核对 29 关节
qpos/qvel 地址及 actuator joint 映射，冻结 low-latency 模型、增益及保护。
所有成对轨迹逐数组一致；没有 DDS、真实设备、额外支撑力或学习更新。

| 资产 / 参考 | 最低骨盆高度 m | 最大绝对 roll/pitch rad |
| --- | --- | --- |
| 官方 29DoF XML / 固定姿态 | 0.1062 | 2.0389 |
| 官方 29DoF XML / 零命令 planner | 0.0698 | 3.0360 |
| 足球资产 / 固定姿态 | 0.7377 | 0.0667 |
| 足球资产 / 零命令 planner | 0.7477 | 0.0783 |

足球资产上的冻结跟踪器能在这两个无球条件下维持站立，不能把接球失败泛化为
“SONIC 连站立都不会”。反之，直接导入 `gear_sonic_deploy/g1/g1_29dof.xml`
仍不能当作已资格原生 sim2sim；官方运行配置实际指向带手的 `scene_43dof.xml`，
还包含初始化/状态传输等其他运行条件。此表不是官方完整系统质量结论，也不
支持直接把 XML 替换进冻结接球考试。

证据：`sonic-stationary-probe-v1/complete.json`。没有据此改写原 M0 或其物理哈希。
