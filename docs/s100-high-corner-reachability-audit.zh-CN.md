# S100：门将高死角可达性审计

日期：2026-08-26

Campaign：`athlete-foundation-v1`

边界：`SIM_ONLY`；本实验只做诊断，不生成候选，不具有晋升权。

## 结论

S98 之后不能继续在同一 residual 拓扑上单纯增加 PPO 时长。S100 在相同高球分布上冻结 S98 actor，对当前控制器家族做了四条配对反事实路线；四条路线在首次运行和重放中都得到 **0 次扑救**，且全部超过安全角速度或出现失败 Episode。

审计决策为：

```text
NEW_LATERAL_LOCOMOTION_DIVE_EXPERT_REQUIRED
```

这不是说“强化学习无效”，而是说明当前网络只能在已有动作拓扑附近做 residual；它没有一个能从球门中央快速并步/交叉步、蹬地、腾空、伸手和落地的完整动作专家。继续让同一个 actor 在这套支架上试，会优先学到更快、更远、但更不稳定的动作。

## 四条路线

所有路线使用：

- 同一个 S98 冻结 checkpoint；
- 相同的两个 seed：`91031`、`91051`；
- 每个 seed 16 个并行环境，即每条路线 32 个 Episode；
- 全部为左右平衡高球；
- 球飞行时间 `0.48–0.62 s`；
- 每条路线完整运行两遍。

| 路线 | 含义 |
|---|---|
| `bounded-parent` | 当前有界 targeted-dive 父控制器，在线 actor 输出为零 |
| `learned-candidate` | S98 冻结 actor 的确定性输出 |
| `full-drive-probe` | 当前控制器家族内把横移驱动与 option gate 提到上界 |
| `full-drive-lunge-probe` | 在上界横移驱动上再叠加 0.8 侧向扑步支架 |

后两条只是“当前控制器家族的预算探针”，不是 privileged oracle，也不能被当作可部署技能。

## 物理结果

| 路线 | 首扑率 | failed rate | 平均最大横移 | 最小手—目标距离 | 最大根角速度 |
|---|---:|---:|---:|---:|---:|
| bounded parent | 0% | 21.88% | 0.730 m | 0.576 m | 7.177 rad/s |
| learned candidate | 0% | 78.12% | 1.203 m | 0.455 m | 7.986 rad/s |
| full drive | 0% | 34.38% | 0.466 m | 0.381 m | 6.690 rad/s |
| full drive + lunge | 0% | 34.38% | 0.468 m | 0.381 m | 6.567 rad/s |

最关键的因果关系是：

1. S98 actor 的确把横移从约 `0.73 m` 增加到 `1.20 m`，说明网络不是完全没学到；
2. 但 failed rate 从 `21.88%` 恶化到 `78.12%`，最大根角速度接近 `8 rad/s`；
3. 把已有 lateral drive 和 lunge 支架推到上界，手离目标更近，但仍然没有任何扑救；
4. 所以短板不是再把一个标量从 0.4 调到 1.0，而是需要新的接触序列和动作拓扑。

## GPU 重放的诚实边界

四条路线的 MJWarp 配对重放在“是否扑救、是否安全、状态是否有限”三类诊断结果上完全一致，因此：

```text
paired_outcome_consistent = true
```

但 GPU 并行物理的浮点归约不是位级确定性的，连续指标存在差异，最大绝对差分别为：

- bounded parent：`0.625`
- learned candidate：`0.732`
- full drive：`0.152`
- full drive + lunge：`0.011`

因此报告同时如实写入：

```text
strict_replay = false
```

S100 不会用“决策一致”冒充“严格重放”，也不会让 GPU 诊断替代 CPU MuJoCo 晋升考试。这里允许决策级一致，只因为实验的结论是“停止当前路线、不要晋升任何东西”。如果未来出现候选，仍必须进入 CPU matched exam。

## 新增工程合同

`training/high_corner_reachability_audit.py` 提供：

- checkpoint 使用 `torch.load(weights_only=True)`；
- G1 asset 资格检查；
- 角色 actor 冻结、无梯度、无候选生成；
- 四条固定路线和成对 seed panel；
- 原始首次指标、完整 replay 指标、最大数值差、位级重放标志、决策级一致性同时留档；
- 输入 checkpoint、locomotion policy、G1 body、kick prior 与实现文件内容哈希绑定；
- `SIM_ONLY`、无硬件命令、无像素评分、非商业演示边界；
- 输出存在即拒绝覆盖。

证据：

- `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s100-high-corner-reachability-audit-v3/audit.json`
- report：`sha256:c83ca69e223bd6e24dcda3ff6df4abdcfaf62d253164ce553393adb9777eed1f`
- S98 actor：`sha256:73c8273ebe32940b841b19c78e683303b3c9608a78f9c391afcea0bacfe52277`
- locomotion prior：`sha256:d1d91b0201beeb649a4624ba40052d10fe4aebe98bf6f4847decf75dd1fee2da`

## 验证

- S100 单元与内容绑定测试：`4 passed`；
- 新模块严格 mypy：`Success: no issues found in 1 source file`；
- 新文件 `ruff check` 与 `ruff format --check`：通过；
- 全量 pytest：`629 passed, 15 skipped, 3 failed`。

全量的 3 个失败仍是外部 S78/S79/S80 历史 evidence 与当前 implementation hash 不一致，和上一轮相同；内容绑定按设计拒绝陈旧证据。本轮 S100 证据已用当前实现重新校验通过。

## 下一阶段：真正的新专家

下一步按 S100 决策执行，而不是再 sweep residual gate：

1. 从 SONIC、MotionDecode 和现有 locomotion prior 中筛选 G1 侧向并步、交叉步、蹬地和落地片段；只把许可证与关节映射合格的来源进入训练集。
2. 先训练不带球的 `Lateral Athlete Expert`：目标是左右 `0.5/1.0/1.5/2.0 m` 指令跟踪、低角动量、双侧对称、末端可继续运动。
3. 在横移专家可达后训练独立 `Dive Expert`，允许 Dream 中摔倒探索，但把接触、角速度、落地和 successor state 留到严格晋升端。
4. 用本体感与接触历史训练 router，在 athlete、dive、absorb、get-up 之间做离散 option 加 150–300 ms 边界融合，避免两个接触动作直接软平均。
5. 只有 actor 真正参与且安全成功的轨迹进入因果 replay；左右高球必须分别达到记忆覆盖门。
6. 最后再接回 S95 动态传球与 S96 高角射门，进行不重置的三 G1 完整 Episode。

S100 的突破不是扑救率已经提高，而是终于用物理反事实证明：继续原路线不会自然长出高死角飞扑；工程应从“残差调参”切换到“运动员横移与扑救专家的动作发现”。
