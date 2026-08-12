# ROSClaw Soccer S2：自由球物理闭环迁出 Core

日期：2026-08-12
范围：CPU MuJoCo、`SIM_ONLY`，不构成真实机器人授权

## 结论

S2 已把 Age-4 标准自由球闭环的领域实现从 ROSClaw Core 迁入
`rosclaw-soccer`，并完成迁移前后逐字节轨迹对照。迁移后的 Soccer runner
独立拥有球场、球门、球网、球物理、自由球技能、SONIC/GR00T 跑动接近、触球
残差、专家路由和足球技能记忆；Age-4 训练入口不再回调 Core 的自由球或足球
Growth 实现。

端到端结果不是近似一致，而是：

- 迁移前后 `trajectory_digest` 完全一致；
- 压缩 NPZ 文件 SHA-256 完全一致；
- `G1FreeKickResult` 结果对象 0 个字段差异；
- 新执行内部的第二次严格重放一致，`strict_replay=true`。

这说明本阶段改变的是模块所有权和扩展边界，没有暗中改变 G1 运动行为。

## 迁移内容

### 足球世界与技能

- `world/field.py`：IFAB 尺寸球场、标准球门、可见球网、足球质量/半径/摩擦、
  球网顺应力和单/双/三 G1 世界构造；
- `skills/shoot/free_kick.py`：单一连续世界中的跑动、衔接、触球、飞行、入网、
  踢后恢复、指标和严格重放；
- `skills/shoot/loft_teacher.py`：只用于训练探针的、有距离门控和力上限的
  task-space teacher。

### G1 provider 边界

- 外部 RoboNaldo、GR00T WBC 和 GEAR-SONIC 资产资格校验；
- SONIC 29 关节闭环 provider 与 GR00T 历史步态 provider；
- MuJoCo 状态/接触适配、关节边界保护、力矩权威投影和技能过渡桥；
- 本地 `G1_DDS_JOINT_NAMES`、硬力矩上限和射门参数合同。

### Soccer Growth

- ballistic contact target/torque residual；
- ballistic skill memory；
- proprioceptive expert router；
- football outcome model。

这些实现保留旧 artifact 的 schema 字符串，以便加载冻结证据；schema 名字是兼容
标识，不代表实现仍由 Core 所有。源码 AST 纯度测试禁止重新 import Core 中的
`football_*`、`ballistic_contact_*`、free-kick、stadium、loft teacher 和旧 G1
MuJoCo backend。

## 严格物理回放

输入完全复用 S1 的请求、支撑链候选、触球 actor 和三个外部模型根目录。资产
资格校验先确认 Body、kick prior、GR00T 和 SONIC qualification hash 与旧请求一致。

新证据：

```text
/code/rosclaw/rosclaw_football/evidence/s2-free-kick-extraction-replay-v1/
```

| 指标 | S1 迁移前 | S2 Soccer runner | 差异 |
| --- | ---: | ---: | ---: |
| trajectory digest | `74425d...74773d` | `74425d...74773d` | 相同 |
| 结果字段差异数 | — | — | 0 |
| 门线目标误差 | 0.0204299277 m | 0.0204299277 m | 0 |
| 感知连续性 | true | true | 0 |
| 踢后后退 | 0.0 m | 0.0 m | 0 |
| 踢后跌倒 | false | false | 0 |
| 饱和 steps | 0 | 0 | 0 |
| 踢后稳定时间 | 3.354 s | 3.354 s | 0 |
| 内部严格重放 | true | true | 0 |

文件 SHA-256：

```text
request.json
63ec33c5df661c98581c1585a6804487e1c4565baf83292b99ae713481d476f8

g1-free-kick.json
132f019359c209516be3b5d7baa15871674c332056815e29c33ee262e9369391

g1-free-kick-trajectory.npz
51b3e890349fdd1b33c427142f375b0c3919ae6a8b659571c1f674a253a6c86a
```

`implementation_hash` 和 `request_hash` 按设计变化，因为实现文件集合迁到了新包；
`stadium_scene_hash` 也按设计变化，因为它显式绑定 builder 源码哈希。Body、kick
prior、SONIC reference 和所有物理结果均保持不变。

## 发现并修复的真实集成问题

第一次新 runner 执行时，最新 Core 已删除旧的
`shared_post_impact_recovery_config()` 工厂函数，Soccer 的恢复桥正确失败关闭。
现在桥接层只依赖 Core 的通用恢复 config/controller API，并在 Soccer 侧显式绑定
冻结的共享恢复参数；单测锁定这些参数，迁移轨迹也证明兼容层没有改变行为。

## 验证矩阵

- Soccer：85 passed，1 skipped；唯一 skip 是只有旧 #271 模块可用时才运行的
  S1 迁移对照测试；
- `ruff check`、`ruff format --check`、`mypy src`、`compileall`：通过；
- Soccer 安装后：CLI、Growth adapter、SimForge task provider 和三个 Dataset
  source 全部被发现，discovery errors 为空；
- 实际卸载 Soccer 后：`rosclaw soccer` 消失，Core 1.2.0 与 `rosclaw simforge`
  正常，三个注册表无 Soccer 残留；
- 无 Soccer 的 Core 定向 CLI/注册表/合同/领域纯度回归：40 passed。

曾启动 Core 全量非 integration/deployment 套件，但它会重复构建约 619 MB 的离线
release bundle。约 5% 时已确认两类与本 PR 无关的既有失败：离线 bundle 安装无法
创建 Python 环境；`tests/agentd/test_operator.py` 的固定审批时间相对 2026-08-12
已经过期。收集完整堆栈后终止，不能把该次全量运行声明为通过。

## 边界与下一步

本 PR 迁移了单 G1 自由球 runner，并已将双/三 G1 球场 world builder 下沉；多人
传球、守门员 runner 和媒体渲染仍应作为后续小 PR 迁移，避免把物理内核、多人
任务和宣传视频混成一个不可审查提交。

当前唯一有意保留的 G1 Core 运行依赖是通用 cerebellar recovery controller。
下一步应把它改为正式 embodiment provider protocol，随后：

1. S2b 迁移 passer/shooter/goalkeeper 和统一媒体渲染；
2. S3 迁移 MotionDecode 足球解释与数据工具；
3. S4 用完整下游栈重新签发 Age-4 canonical evidence；
4. Core 删除已无下游调用者的旧足球模块；
5. 架构收口后进入 Age-5 First Touch 连续 episode。

视频是物理证据的下游展示，不参与 promotion truth；本阶段没有提交 MP4、原始轨迹
或模型权重。
