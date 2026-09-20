# Human Tactical Prior：事件意图与 waypoint 联合学习

2026-09-21，继 waypoint 先导实验后的第二个固定预算实验。不是 G1 身体策略，
也尚未完成 Tactical World 或 MAPPO 闭环。

数据继续使用 [SkillCorner Open Data](https://github.com/SkillCorner/opendata)，
固定提交 `4340d274572876239c154c90bc507a9b3250a656`，MIT。
训练比赛为 2017461、1996435，开发比赛为 2006229。整场隔离，未使用封存数据。

将官方动态事件按比赛、半场、球员、队伍和 `[start,end)` 帧区间连接到 tracking。
冲突标签剔除，未知标签不当成 idle。只有四类具有本次明确映射的标签：

| 训练意图 | 官方事件映射 |
| --- | --- |
| SUPPORT | off-ball run: support / coming_short |
| ADVANCE | run_ahead_of_the_ball |
| RUN_BEHIND | behind |
| PRESS | pressing / pressure / counter_press / recovery_press |

COVER、RECOVER 暂未建立可靠映射，不能宣称六类齐备。标签是回顾性事件注释，
只作监督目标；输入没有未来事件结果或未来阶段。特征与前一先导实验相同。

最终得到 7,350 条训练、3,373 条开发带标签样本。共享两层 64 宽 tanh 编码器，
分别预测下一半秒 waypoint 修正及四类意图；固定 12 epochs、batch 256。
预先固定三个种子，全部保留最终模型，不按开发分数挑选种子或 checkpoint。

| 模型 / 种子 | 意图 macro-F1 | accuracy | waypoint 平均误差 |
| --- | ---: | ---: | ---: |
| 多数类 PRESS / 匀速位移基线 | 0.1967 | 0.6487 | 0.51255 m |
| 9202026 | 0.6223 | 0.8206 | 0.45685 m |
| 9202027 | 0.6558 | 0.8109 | 0.45599 m |
| 9202028 | 0.6737 | 0.8251 | 0.45663 m |

三份 checkpoint 独立重载后，意图 logits 和 waypoint 输出逐元素相同。
RUN_BEHIND 仍是弱项，三个种子的 F1 约为 0.228、0.450、0.446，常与 ADVANCE
混淆。不能用较高总体 accuracy 掩盖该弱项。

本次只评估有明确事件标签的子集，与前次全量 waypoint 的 0.27821 m 不可直接比较。
仅一场独立开发比赛，时间相邻样本高度相关，三随机种子不是三场独立比赛。
本结果证明可从公开职业足球轨迹学出部分意图相关结构，不证明因果决策质量、
机器人执行能力或团队闭环收益。

下一步仍需非 29DoF 的 2v1 Tactical World，并接入物理实验估计的技能能力；
未测得的接球概率、技能时长、执行后 readiness 必须保持未知，不能手填乐观数值。

外置证据：`receiving-mechanism-reboot/tactical-intents-v1` 保存数据合同，
`tactical-intent-prior-v1` 保存三组权重、开发预测、完整混淆矩阵及训练记录。
准备与训练脚本保存在同一实验根目录，模型未进入任何真实机器人执行链路。

## 可复用模型接口

`training.human_tactical_prior.HumanTacticalPrior` 提供不依赖 Torch 的 NumPy 推理。
导出限定固定架构、20 维特征合同与四类意图顺序，绑定 checkpoint、数据合同和
导出文件哈希，拒绝 pickle、非有限参数、错误形状和缺失的角色/球权类别。
输出为进攻坐标系的半秒位移建议与原始 logits，不是校准概率，也不是执行许可。
未经评估不允许把人类球场模型直接缩放到机器人小场。

三模型在 3,373 条开发样本的导出验证中，全部意图类别与 Torch 原预测一致；
NumPy 位移与原预测最大绝对差小于 `3.54e-7 m`，不宣称跨数值后端逐位相同。
证据在 `tactical-intent-export-v1/verification.json`。此导出没有追加训练。

## 冻结模型追加整场验证

在读取新增比赛结果前固定 2006363、2007448 两场，仅作追加公开开发验证。
复用原特征与标签程序，跟踪文件按官方 LFS 哈希核对，不更新模型、选种子或
改变分类阈值。两场分别有 3,914、3,157 条有明确标签的样本，4 条冲突剔除。

| 新开发比赛 | 匀速 waypoint 误差 | 三种子 waypoint 误差 | 三种子意图 macro-F1 |
| --- | ---: | ---: | ---: |
| 2006363 | 0.49659 m | 0.44349–0.44390 m | 0.6630–0.7024 |
| 2007448 | 0.49812 m | 0.45161–0.45193 m | 0.6856–0.7257 |

两场各三个种子都改善位移预测；多数类 baseline 的 macro-F1 分别为 0.1837、
0.1793。RUN_BEHIND 依然显著弱于 PRESS。现在共三场独立开发比赛，而不是
只有一场；仍不属于封存考试，不能证明比赛闭环、机器人迁移或联赛收益。
新增证据在 `tactical-external-dev-v1`，含先行合同、哈希、预测及全部混淆矩阵。
