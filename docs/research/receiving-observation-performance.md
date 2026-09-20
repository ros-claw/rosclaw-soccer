# 接球状态观测的等价提速与回归边界

2026-09-21，CPU MuJoCo，SIM_ONLY。这里只优化观测开销，不修改动作、
奖励、物理模型、考试判据或训练权重。正在运行的旧版本 M0 实验保持原代码。

## 先测量，再改实现

对固定 blue.defender 已知成功课程进行 cProfile，仿真 6 秒，在第 30 控制帧
记录支撑状态。8 个球员各自通过完整持久检查点复制状态，导致同一个大型
编译模型被序列化和哈希 17 次：8 次 capture、8 次 restore，以及一次真正
需要保存的初态检查点。支撑观测累计约 18.19 秒，成为主要瓶颈。

`observe_support` 改为在已确认 `data.model is model` 的同步调用内，用
相同 `mjSTATE_INTEGRATION` 规格复制到新的 MjData，再刷新派生观测。
原状态、接触和求解器 warm-start 不被改写；非有限积分状态仍拒绝。
这不是外部检查点恢复。持久 `PhysicalCheckpoint` 的模型、版本、状态哈希
验证完全保留，也没有缓存可变模型的旧哈希。

| 检查 | 原实现 | 新实现 |
| --- | ---: | ---: |
| 完整编译模型哈希调用数 | 17 | 1 |
| cProfile 记录的总时间 | 33.523 s | 15.455 s |
| A0 原始数组一致性 | 参考 | 189/189 精确相同 |
| A3 原始数组一致性 | 参考 | 205/205 精确相同 |

这是单次配对的性能诊断，机器同时有其他任务，不能据此承诺所有训练都提速
同样倍数。显式功能检查验证了实时状态不变、与检查点往返等价、无需模型
序列化和非有限输入拒绝。相关模块组合测试 89 项通过，聚焦 Ruff/mypy 通过。

外置证据：`receiving-mechanism-reboot/m0/adapter-profile-v1/` 和
`support-copy-qualification-v1/`；后者含 A0/A3 独立重放核对。

## 补充容易样本，不改原始 64 状态考试

选择预先已知成功的 historical-000 至 historical-007，每个角色一例，
球速 0.75 m/s、侧偏 -0.08 m。固定第 30 帧测量，共 16 次原生/独立重放。
8 例全部保留原生成功，球位、球速和焦点关节数组与历史记录精确相同，
新增两次重放的所有数组也精确相同。8 个入口均在第一次脚部及非脚部触球前。

这些入口都为右足支撑，测得右足地面法向力约 270–281 N；不能把它们称为
覆盖所有支撑相位或完全静止状态。它们是已知容易样本的补充证据，不计作
新泛化成功，也尚未完成四动作接口搜索。原 `precontact-bank-v1.json` 不变。
证据：`m0/positive-control-states-v1/complete.json`。

## 全量测试没有全绿，保留失败

本轮默认 CPU pytest：4,746 passed、34 skipped、11 failed。
11 项失败来自旧 S78/S79/S80/S93/S104/S106/S107/S108/S113/S114 的外部
证据/视频权威校验。在本轮修改前的 a3ba8a3 工作树上，单独重跑相同相关
文件得到完全相同的 11 项失败（其余 48 passed、3 skipped）。
S104 已明确确认旧证据绑定的 implementation hash 与当前实现不同。
没有覆盖旧证据、重签哈希或放松 validator；不能声称全库测试已通过。
核对凭据：外置 `regression-artifact-baseline.xml`。

默认测试中，部分实际资产测试按环境配置跳过；本轮另行配置真实 G1 资产、
SONIC 模型与冻结 R0 检查点运行 S199、S200 和 SONIC latent 相关集成测试：
21 passed、0 skipped，48 条 TorchScript 弃用警告。凭据为
`physical-integration-regression.xml`，不能用默认 skip 冒充物理验证。
