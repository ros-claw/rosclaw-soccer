# 随球击打实验 v35–v36：开发集改善未迁移

`SIM_ONLY` 实验，未晋升为球队技能，更没有宣传级连续比赛。

v35 在 v34 已消费的 8 个发生脚触球场景上，比较前向随球 8 cm 与 16 cm。原版仅 1/8 有效回球；8 cm 为 3/8、1 次不安全，16 cm 为 3/8、0 次不安全，按冻结规则选 16 cm。v35 报告 `/data/rosclaw_overflow/rsi-team-strike-follow-through-v35/report.json`，哈希 `sha256:71a6ad0b96b1cd9a2f016b98b7b589a2101a681326ab5dd8a9543e1ee43d2d14`。这些球况早已用于发现弱击球，不能作独立验证。

v36 新球况按 Parent、零随球、16 cm 随球三方配对，计划 24 场；第 22 场 Parent 提前终止，旧收集器直接抛异常，未形成完整验收。对**已完成的前 21 场**逐个重验 episode 报告、物理轨迹和动作轨迹哈希，审计结果：Parent 0/21 有效、3 次不安全；原版 **5/21 有效、1 次不安全**；16 cm **4/21 有效、3 次不安全**。其中 16 cm 有三场新增安全回归，且损失原版一场原本 `0.941 m/s` 的有效回球；不能把 v35 的两场改善外推。部分审计 `/data/rosclaw_overflow/rsi-team-strike-follow-through-partial-audit-v36/report.json`，哈希 `sha256:78def97ef5be2a7af7b2bd843f96373c1646868c3ac847a460af94c5955aef71`；明确 `fresh_exam_complete=false`、`gate_passed=false`。

收集器已改为 Parent 提前终止时整组保守记失败，动作审计拒绝也记候选失败；不得悄悄跳过难例。工程判断：统一加大击球幅度伤害稳定性，单一参数曲线不适合全部来球。下一阶段需要大量*完整配对*不同球况，学习何时选择导航/击打，并检验安全和出球的双目标泛化；不能通过剪辑个别成功镜头来证明球队已成长。
