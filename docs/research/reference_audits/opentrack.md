# OpenTrack：借鉴学生访问状态上的教师标注，不提前启动 DAgger

2026-09-21。本地官方仓库固定到
`cb9b751993a2483e5d1805a2565ddbfe950c04c9`，检查时工作树干净；代码许可 Apache-2.0。
本次为实现审阅，没有启动其训练、部署或设备接口，没有复制模型/数据进 Soccer。

来源：[官方仓库](https://github.com/GalaxyGeneralRobotics/OpenTrack)。
主要检查 `track_mj/learning/policy/dagger/dagger_horizon.py`、
`utils.py`、README 和 `storage/training_configs/dagger/demo.json`。

## 与当前问题直接相关的机制

- 每一步从同一个当前环境状态获取学生观测和 privileged teacher observation。
  教师按 motion cluster 路由，但真正推进环境的是 **学生动作**，不是教师轨迹重放。
- 在这些学生实际访问状态上保存教师动作，使用短 horizon 的采集批次更新模型。
  本次读到的循环不是一个跨所有历史迭代永久累加的数据仓，不能据名称臆称已有
  无限 DAgger replay buffer。
- 当学生输出绝对 motor target 而教师输出 residual 时，显式通过环境的目标映射
  转换教师标签。ROSClaw 同样不能把额外任务力矩直接当作 ±0.1 rad 残差标签。
- 记录每步之后的 episode done，用于相邻动作正则的有效边界。未来 recurrent
  接球学生还必须保留 episode、角色、基础模型历史和 reset mask，不能将相关帧
  打乱成“独立场景”或在跨重置序列间传播隐藏状态。
- 上游示例按训练损失保存 best checkpoint。Soccer 只能将其视为候选选择信号；
  必须另过物理接球、历史保留、独立种子和一次性 sealed 门，不能直接晋升。

## 为什么这不是当前启动学生的理由

上游 README 对 generalist v1/v2 的解释强调了 specialist teacher 的覆盖质量：
教师没有掌握的困难动作不能靠蒸馏自动补齐。它讨论的是动作跟踪，不能转述成
已解决我们的球—脚接触、卸力或 Receive-to-Ready。

当前有界接球反馈教师仍未达到 E2。先把真实失败状态上的教师能力做出来，再
接入 DAgger；本审阅不授权绕过总纲的 E1/E2 门控。也不运行会覆盖原始动作文件
的上游预处理程序。外部数据保持只读，转换应输出到独立派生目录。
