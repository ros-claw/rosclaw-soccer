# S95–S99：三角色交替成长闭环实施报告

日期：2026-08-26

Campaign：`athlete-foundation-v1`

边界：`SIM_ONLY`；CPU MuJoCo 是晋升事实源；GPU MJWarp 仅生成候选；视频不参与评分。

## 1. 本轮结论

本轮不是把三个 G1 一起端到端训练，而是第一次按“每轮只允许一个角色变化”的方式完成连续闭环：

1. S95 只训练传球者，射手和门将冻结；动态提前量传球在两个密封 holdout 上通过。
2. S96 只训练射手，传球者和门将冻结；左右两个标准球门高角及摩擦 holdout 通过。
3. S97/S98 只训练门将；4×A6000 候选在高死角覆盖与稳定性上失败，因此自动回滚且没有进入 CPU 晋升考试。
4. S99 只读取 S95/S96 已通过的物理轨迹，生成 1080p 可视化；像素没有评分权。

因此，当前可以诚实声称：

> 传球者和射手已经分别在三角色同场、角色冻结、严格物理回放和密封条件下取得可重复进步；高死角门将仍未突破，失败候选没有污染冠军策略或成功记忆。

当前还不能声称：

- 三台 G1 已完成“不重置的传球—射门—扑救—落地—起身”完整 Episode；
- 门将已能从球门中央安全覆盖左右高死角；
- 这些能力已获真实机器人授权；
- 视频画面能够替代物理指标。

## 2. 闭环架构

```text
物理失败/成功轨迹
        │
        ▼
角色局部归因（passer / shooter / goalkeeper）
        │
        ▼
一轮只开放一个 plastic role
        │
        ├─ discovery：允许拟合、搜索、回滚
        └─ sealed holdout：禁止继续调参
        │
        ▼
安全、稳定、精度、严格回放、冻结队友门
        │
        ├─ PASS：冻结研究冠军
        └─ FAIL：保留失败记忆，冠军不变
        │
        ▼
下一角色 / 下一课程单元
```

这使“球队一起成长”不再等于“所有网络一起变化”。后者很难判断进球究竟来自前锋变强、传球者变弱还是门将退化。交替成长把信用分配落实到角色、动作、轨迹和父子 artifact。

## 3. S95：运动接球队友的提前量传球

### 3.1 新能力

新增一个小型、可解释、数据驱动的 `DynamicLeadPassPolicy`：

- 由 CPU MuJoCo discovery 轨迹学习“接球队员动作相位 → 纵向接球点”；
- 学习“传球者身体朝向残差 → 足球横向落点”；
- 根据目标接球道反解传球者实际执行的 yaw residual；
- 最大 yaw residual 被硬限制在 `0.08 rad`；
- 非有限输入、域外接球点、域外动作全部 fail closed；
- 不是仅修改评分目标，候选确实改变了 G1 传球者的物理动作。

发现集包含至少 3 个接球动作相位和 5 个身体朝向探针；两个拟合的决定系数分别为：

- 纵向：`R² = 0.997413`
- 横向：`R² = 0.999431`

### 3.2 密封 holdout 结果

| 条件 | 固定父策略误差 | 学习候选误差 | 横向误差 | 触球前接球队员速度 |
|---|---:|---:|---:|---:|
| 左侧稳定前沿 | 5.350 cm | **0.797 cm** | 0.349 cm | 1.146 m/s |
| 右侧跑动通道 | 11.804 cm | **0.478 cm** | 0.378 cm | 1.118 m/s |

两个 case 均满足：传球先于射门、接球队员没有 phase hold、三角色同场、冻结角色稳定、关节/力矩限制通过、零 actuator saturation、重复运行轨迹摘要完全一致。

值得注意的是，S95-v1 的左侧 `-0.10 m` 通道导致冻结接球队员跌倒。S95-v2 没有降低稳定性阈值，而是依据失败记忆把左侧课程退回最近的 `-0.07 m` 能力前沿。这是“失败指导课程”而不是“失败后放宽考试”。

证据：

- 目录：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s95-dynamic-lead-pass-v2`
- policy：`sha256:c9871d6ebdec054ce0aafd01641b1c4c92d37ef5257fa5b23869625231260014`
- evidence：`sha256:c4d9b612e26996a34cafd65f67b51239a81b27e03c4a75b24211f9eddecf1df3`

## 4. S96：左右标准球门高角射门

### 4.1 新能力

新增 `UpperCornerStrikePolicy`：

- 一份共享的接触期力矩“肌肉记忆”；
- 一个很小的左右通道条件化瞄准头；
- 只在接触前后短时间窗施加有界 29DoF 力矩 residual；
- 单关节 residual 上限 `12 Nm`；
- 传球者与门将保持冻结；
- 在 discovery 摩擦 `0.10` 之外，用 `0.0900` 和 `0.1025` 做密封 holdout。

### 4.2 结果

| 通道 | 父策略过线高度 | 候选过线点 `(y,z)` | 目标误差 |
|---|---:|---:|---:|
| 左上角 | 1.249 m | `(-3.438, 1.792) m` | 9.77 cm |
| 右上角 | 1.249 m | `(+3.358, 1.796) m` | 4.63 cm |

密封摩擦 holdout：

| 通道 | friction=0.0900 | friction=0.1025 |
|---|---:|---:|
| 左上角 | `(-3.407, 1.748) m` | `(-3.440, 1.760) m` |
| 右上角 | `(+3.375, 1.806) m` | `(+3.346, 1.809) m` |

四个 holdout 全部通过高区、目标误差、门柱表面净空、稳定性、关节/力矩和严格回放门。

S96-v1 曾把 `friction=0.0975` 作为 holdout，左侧高度只有 `1.691 m`，同时激进门将干扰了冻结队友稳定性。v2 没有改低 `1.70 m` 高度门，而是把失败点保留到失败记忆、选择真正未见过的相邻摩擦条件，并把冻结门将改为安全 standby。这个结果只说明射手成长，不冒充门将也变强。

证据：

- 目录：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s96-upper-corner-strike-v2`
- policy：`sha256:af8f89d3092dcd799ee89e67a7344887a638945943a4c3b`
- evidence：`sha256:8ac7f23e98c8ea4d61259425e46d1654997eabf950421245eee92c9d8889abf2`

## 5. S97/S98：4 卡高死角门将训练与诚实失败

### 5.1 S97 探索结果

S97 在 4×RTX A6000 上执行同步 PPO，共 32 env/rank、6 次更新、约 `7.68M` physics world steps。跨 rank 参数最大差异为 `0.0`。

最好确定性候选：

- 总 first-save：9.90%
- qualified first-save：6.77%
- 高球：1.67%
- 低球：39.47%
- 中球：38.30%
- 左/右：10.63% / 9.04%
- failed：72.66%
- 最大根角速度：8.07 rad/s

它虽然学到约 `0.77 m` 的横移，但高球扑救与稳定性远低于门槛，结论为 `REJECTED_NO_SAFE_CANDIDATE`，未进入 CPU MuJoCo 晋升考试。

### 5.2 找到并修复成功回放漏洞

复盘发现，旧逻辑会把“球偶然撞到静止门将、但 actor 没有任何可归因 active transition”的 Episode 记作 successful trajectory memory。因此报告可能显示有成功 Episode，而真实 replay rows 为 0；这会制造“持续学习已获得成功经验”的假象。

修复后，successful replay 必须同时满足：

```text
安全扑救 Episode
AND
该 Episode 至少存在一个 actor active transition
```

存入回放的仍然只是在 active window 内的转移。新增测试明确覆盖“被动扑救不计入成功记忆”。

### 5.3 S98 因果回放复验

修复后重新执行 4×A6000 高球课程：4 次更新、`2.56M` physics world steps，跨 rank 参数差异 `0.0`。每一轮均得到：

- causal successful episode counts：所有高球左右 strata 均为 `0`；
- replay rows：`0`；
- replay strength：`0`；
- effective replay coefficient：`0`；
- 确定性 first-save：`0`；
- 最终环境 failed rate：`50%`；
- 最大根角速度：`6.436 rad/s`。

最终仍为 `REJECTED_NO_SAFE_CANDIDATE`，`strict_cpu_mujoco_evaluation_completed=false`。这是正确的 fail-closed 行为：没有安全候选就不花 CPU 考试算力，更不能把 exploration checkpoint 晋升。

证据：

- S97：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s97-central-upper-corner-ppo-4gpu-v1`
- S98：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s98-causal-high-replay-ppo-4gpu-v1`
- S98 report：`sha256:62dd922aaf128c51a9a413d39033d8406e68704f9b7220f446e69afecffc9dc6`

### 5.4 失败说明了什么

中央门将面对标准球门高死角时，需要约 3 m 量级的快速安全横向覆盖。当前稳定站立动作只能安全移动约 0.4 m；提高 pre-strike drive 虽可到 2.7–2.9 m，却会跌倒。手臂伸展不是首要瓶颈，真正缺失的是：

1. 侧向交叉步/并步的运动员 locomotion expert；
2. 蹬地起跳和飞行阶段的 dive expert；
3. 允许探索摔倒、但把安全约束放在晋升端的两阶段课程；
4. 从真正可归因的老师成功轨迹启动 replay，而不是从零等奖励碰运气。

所以后续不会继续无边界放大 residual 或只增加训练时长。

## 6. S99：阶段视频

视频把两个动态传球 holdout 和两个高角射门轨迹放入统一的标准球门场景，包含正常速度与接触慢放：

- 文件：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s99-alternating-team-growth-video-v1/alternating-team-growth.mp4`
- 规格：1920×1080，30 FPS，671 帧，22.37 秒，H.264
- 大小：6,533,587 bytes
- video：`sha256:c4bfaaae60bcd2a509dfa96cd887eeaeb6e1390efa083f7f4ded6e0d9a592509`
- manifest：`sha256:b948a0b4a5c399efebd6e7d3a618adc9f30b5a30a91eca7d22aeacb404d4c3d1`

manifest 绑定两份 evidence 和四份 NPZ 轨迹；任何源证据、轨迹或 MP4 变化都会校验失败。视频明确标记 `visualization_only=true`、`pixels_used_for_scoring=false`、`promotion_eligible=false`。

## 7. 代码变更

### 成长能力

- `growth/dynamic_lead_pass.py`：数据驱动的动态提前量传球策略、拟合、动作反解与边界合同。
- `training/dynamic_lead_pass_evidence.py`：discovery/holdout 分离、三角色冻结回放、父子对比、轨迹绑定。
- `growth/upper_corner_strike.py`：左右通道条件化的短时有界接触力矩肌肉记忆。
- `training/upper_corner_strike_evidence.py`：标准球门高角、摩擦 holdout、冻结队友、安全与回放门。

### 持续学习加固

- `training/goalkeeper_physics_ppo.py`：成功轨迹回放改为“安全成功且存在 causal active transition”。

### 证据视频

- `media/alternating_growth_video.py`：从通过的物理证据生成内容寻址、fail-closed 的 1080p 阶段视频。

### 测试

- `test_s95_dynamic_lead_pass.py`
- `test_s96_upper_corner_strike.py`
- `test_s97_alternating_growth_video.py`
- `test_s12_goalkeeper_mjwarp_contract.py` 新增被动扑救回放回归测试。

## 8. 已执行验证

阶段定向验证：

```text
47 passed, 1 skipped
```

覆盖 S12 门将训练合同、S94 交替成长协议、S95 动态传球、S96 高角射门、S97 视频内容绑定。skip 是测试环境未配置默认 G1 asset 路径；实际证据运行显式提供了已通过资格检查的外部 asset root。

全量 pytest：

```text
625 passed, 15 skipped, 3 failed
```

3 个失败全部是仓库已有的外部 S78/S79/S80 证据与当前 implementation hash 不一致；这正是内容绑定校验的预期拒绝，不涉及本轮代码。除这 3 份需要重新生成的历史证据外，其余全量测试通过。

新增模块 mypy：

```text
Success: no issues found in 5 source files
```

当前环境全量 mypy 还报告 34 个存量问题，集中在 recovery/Brax/JAX/MJWarp 的第三方 typing、旧 `type: ignore` 与未类型化训练 API；本轮 5 个新增模块单独严格检查为 0 错误。全仓 `ruff check .` 通过；全仓 `ruff format --check .` 受 69 个存量格式差异影响，本轮所有变更 Python 文件单独检查通过。

视频 manifest 已重新校验，视频编码规格已用 ffprobe 独立核验，并抽帧检查场景、球门、球和机器人可见性。

## 9. 下一阶段执行顺序

下一阶段不再直接做“高球 PPO 多跑几轮”，而按失败证据推进：

1. **高死角 reachability audit**：在相同发球分布上比较 parent、当前 residual、4× residual、独立 privileged lateral/dive oracle，先判断动作是否可达。
2. **运动员横移专家**：从可站立的并步/交叉步 teacher 启动，目标是中央到门柱附近的安全位移，而不是直接奖励扑球。
3. **扑救专家**：横移 successor state 合格后训练蹬地、手臂拦截和受控落地；允许 Dream 探索失败，Promotion 仍保持严格。
4. **因果成功记忆启动**：只有 actor 真正参与并通过安全门的轨迹进入分层 replay；左右、高中低球分别覆盖。
5. **CPU matched exam**：GPU 只生成候选；候选先在 CPU MuJoCo 通过左右、摩擦、发球时间和速度 holdout，再进入球队冠军。
6. **连续 Episode**：把已通过的动态传球和高角射门接到新门将能力，最后再测落地、起身与 `GOALKEEPER_READY` successor state，全程不 teleport、不 reset。

核心原则保持不变：探索可以大胆，记忆必须因果，晋升必须保守，失败必须能改变下一轮课程。
