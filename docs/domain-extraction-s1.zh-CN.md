# ROSClaw Soccer S1：足球 Growth 垂直切片迁出 Core

日期：2026-08-12
范围：CPU MuJoCo、`SIM_ONLY`，不构成真实机器人授权

## 结论

本阶段把 Age-4 当前实际依赖的第一组足球 Growth 实现迁到
`rosclaw_soccer.growth`，并让训练入口不再直接 import Core 中同名足球模块：

```text
approach_strike_contracts
approach_strike_residual
phase_conditioned_residual
ballistic_contact_impulse_actor
football_motion_prior
```

同时把 Age-4 使用的 no-pickle IQL 推理子集和 G1 关节顺序放入显式的
`rosclaw_soccer.providers.g1` 层。它们以后可继续迁到独立 Unitree provider，
但不再因为“用了学习算法”而伪装成任务无关 Growth Core。

这次是 strangler 式迁移的第一刀：#271 中旧模块暂时保留给尚未迁移的
SimForge/CLI 调用者，Soccer 已切到下游实现；等 S2 把 free-kick world 和
runner 迁出后，才能删除 Core 旧实现，避免双写。

## 插件边界

Soccer 现在声明：

- `rosclaw.cli_extensions` → `rosclaw soccer ...`；
- `rosclaw.growth.adapters` → `soccer.growth`；
- `rosclaw.simforge.tasks` → `soccer.academy`；
- `rosclaw.dataset.sources` → `soccer.motiondecode`、`soccer.omnicontact`、
  `soccer.g1-retargeted-motions`。三个上游各自绑定来源 URI 与本地下载 revision，
  不把不同授权、版本和数据语义混成一条来源声明。

三个 revision 来自本地 Hugging Face 下载缓存的 metadata，而不是把 `main` 当作
不可变版本；descriptor 只声明来源，Dataset Doctor 仍需对实际文件形成 inventory/
hash receipt。数据可读不等于获得训练或再分发授权，许可证门保持独立。

Growth adapter 将 Soccer 的 reward/cost/failure 语义写入 task-neutral
`ExperienceSegment`。unsafe-negative 的踢后失稳被诊断为
`soccer.post_contact_instability`，但 adapter 不能激活 candidate 或获得硬件权限。

SimForge provider 只描述 `soccer.age04_regulation` 与
`soccer.first_touch`；discovery 不创建 MuJoCo。Dataset source 只收到
`dataset_id + relative_path`，不会得到数据根目录、文件内容、runtime 或 driver。

## 迁移一致性

迁移保留了原 artifact schema 字符串，这是刻意的兼容策略，而不是仍属于
Core 的证据。冻结的旧实现与新实现对同一输入验证：

- contact actor `to_dict()` 完全一致；
- actor hash 完全一致；
- experiment context hash 完全一致；
- approach residual config hash 完全一致；
- phase-conditioned residual config 内容完全一致。

迁移后的测试复制了原 contact actor、football motion prior 和 phase residual
测试，不只测试 wrapper。

## 严格物理重放

使用 Age-4 v3 的冻结 actor、支撑候选、原配置与产生证据时的 Core 实现
`dae08eee`，从迁移后的 Soccer 类重新执行了完整 teacher-free MuJoCo replay：

| 指标 | 结果 |
| --- | ---: |
| strict replay | true |
| migrated actor executed | true |
| 门线目标误差 | 0.0204299277 m |
| 执行器饱和 | 0 steps |
| 踢后后退 | 0.0 m |
| 踢后摔倒 | false |
| 硬件命令 | false |
| Soccer Age-4 六轴评审 | PASS，failure codes 为空 |
| 更严格 Core 真死角 aggregate | false（与迁移前一致） |

新证据位于：

```text
/code/rosclaw/rosclaw_football/evidence/s1-extraction-replay-v1/
```

文件 SHA-256：

```text
g1-free-kick.json
f43640d5bac512f1dbcf1d98bbbfbd6475f6db94f9b9fde7c30ca1da72c3ef70

g1-free-kick-trajectory.npz
51b3e890349fdd1b33c427142f375b0c3919ae6a8b659571c1f674a253a6c86a
```

第一次尝试用 #271 的更新后 HEAD 重放时，Core 正确返回
`contact impulse actor experiment context mismatch`。原因是后续版本新增
`football_motion_prior_velocity_blend`，旧 actor 的 context commitment 不包含该
字段。切换到产生原证据的 `dae08eee` 冻结实现后才允许执行。这证明版本绑定
不是装饰：新代码不能悄悄冒充旧 actor 的执行上下文。

## 安装/卸载闭环

在包含 ROSClaw #305/#307/#308/#310/#311 的临时组合树中：

- Soccer cross-repository tests：58 passed、1 skipped（仅迁移等价性测试需要旧实现）；
- entry-point discovery：Growth/SimForge 与三个独立 Dataset source 均被发现，
  errors 为空；
- 实际卸载 `rosclaw-soccer` 后，Core 不再显示 soccer 命令；
- 无 Soccer 的 Core Growth/Dataset/SimForge/CLI/纯度回归：67 passed；
- 重新安装后 `rosclaw soccer academy status` 恢复并报告 `SIM_ONLY`。

Soccer 在当前公开 Core 依赖上的自身测试为 40 passed、7 skipped；跳过项均明确
绑定尚未合并的跨仓库合同或只在迁移审计时需要的旧实现。使用堆叠 Core 后，
除旧实现等价性审计外，其余 58 项全部执行通过；另在冻结旧实现上单独验证
迁移等价性 1 passed。

## 下一步

1. S2 迁移 `g1_free_kick_showcase`、球场/球门/足球、loft teacher、多人任务和视频；
2. 用 provider protocol 替代 Soccer 对 Core `unitree_mujoco_backend` 的临时依赖；
3. 在 Core 删除已无调用者的 football Growth/SimForge/CLI 文件及测试；
4. 无 Soccer 安装运行 Core 全量 CI，安装 Soccer 后重放 canonical Age-4；
5. 完成收口后进入 Age-5 First Touch 连续 episode，而不是继续在 Core 增加射门特例。
