# 完整接球序列：进展与短视规划反例

2026-09-21，CPU MuJoCo，SIM_ONLY。E1/E2 未完成资格验收，没有学生训练、
PPO 更新、盲测开封或晋升。总纲仍在实施。

从六份已认证成功四节点提案重建动作库，绑定原始轨迹、成功执行、增益及
合同。目标排除本角色在红蓝两队的数据，后卫可用六份、中场四份。它们仍
是已见研究课程，排除角色不等于盲测。固定来源按预声明顺序取 source_01，
没有按目标结果挑选。

`receiving_desired_plan` 展开连续 50 Hz 期望偏移，预测和实际执行均取对应
行，不保持第一行。原幅度、滤波、步进及力矩保护不变。30 次私有预测通过
恒定序列精确退化、时变滤波、输入非干扰和重复一致性检查。

两案例均为 0.75 m/s、侧偏 -0.08 m；附加接触教师在第 30 帧后被抑制。

| 条件 | blue.defender | blue.playmaker |
| --- | --- | --- |
| 原生残差、无附加教师 | 失败 | 失败 |
| 固定完整序列 source_01 | 成功 | 成功 |
| 每 100 ms 用 500 ms 预测择优 | 失败 | 成功 |
| 首次按相同预测择优，此后锁定整段 | 失败 | 成功 |

前三行 12 次执行，末行另四次，不是 16 个独立课程。全部安全、重放精确。
固定序列不是闭环教师。后卫本角色四种教师种子此前均失败，其他角色序列
却能成功，进一步说明没搜到不等于动作接口做不到。

首次 MPC 均选 source_06；锁定后仍一成一败。首次候选、分数、选择和之后
五帧物理前缀与原 MPC 精确一致，后卫失败至少包含初始排序不足。0.60 s
查询只预测到 1.10 s，尚未覆盖约 1.30 s 的触球；source_01/source_06
评分约 2.4951/2.4947，这个短期优势没有预测完整结果。在线后卫最终一次
脚后碰腿，尾段最远脚距约 0.615 m、最大球速约 1.036 m/s，原考试失败。

后续六课程对照保持未预测违反约束的 incumbent，不为微小评分差换方案。
除两份既有课程，每角色增加 (0.65 m/s, -0.12 m)、(0.85 m/s, -0.04 m)。
完整预检后执行，预算 24 次；须保留既有成功、无安全回退，并超出固定
先验，才算该次 cheap-falsification 通过。结果以 completion 为准。

[Hindsight Plan](https://arxiv.org/abs/1609.09001) 讨论短时域 MPC 局限及
利用离线长时域计划改善目标；[Contact-Implicit MPC](https://arxiv.org/abs/2107.05616)
围绕参考轨迹推理接触时序与力。只借鉴检查思路，不据此宣称本实验原因
已证明，也未移植优化器或新增 RL 算法支线。

本轮三成功、一失败均经轨迹认证、重放和原考试重算，再走 Core Practice
record → strict verify → distill → 本地 ingest → query。查回 3/1，重复
导入失败仍一条；预测与实测分开。这是经验保留，不是模型已学会。

Soccer `7fe284e` 全量 **4994 passed、55 skipped、11 failed**，XML 比较
仍为 `e0cfc48` 相同十一项历史失败，无新增；不重签历史、不称全绿。
相关 97 项测试通过。Core `02564608` 远端自动测试全部通过，人工待审。

证据相对 `receiving-mechanism-reboot`：

- `m0/plan-predictor-equivalence-v2/complete.json`：30 次预测检查。
- `m0/sequence-transfer-teacher-v2/complete.json`：12 次实际执行。
- `m0/sequence-commitment-diagnostic-v2/complete.json`：四次锁定诊断。
- `m0/sequence-continuity-teacher-v1/`：六课程后续对照。
- `practice-sequence-transfer-memory-v1/complete.json`：3 成功、1 失败记忆。
- `reboot-regression-7fe284e.xml`：全量测试。

集成失败保留：预测器 v1 因旧 Core 无新模块在物理前导入失败；序列教师
v1 完成四次原生对照后，新提案因漏合同字段被拒绝，未执行；锁定诊断 v1
首次执行后以 tuple/list 直接比较 JSON 回读结果而中止，未在断言前保存
轨迹，该次不计成败。v2 改用规范内容哈希并先落盘再断言，旧源码和各
STOPPED.md 均保留。
