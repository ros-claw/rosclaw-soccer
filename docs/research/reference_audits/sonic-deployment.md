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
