# 中间侧向增益 v293：仍触膝，开发门拒绝

状态：**36/36 次独立物理回放已审计 / 开发门失败 / SIM_ONLY**。
预注册协议 `docs/rsi/protocols/approach-intermediate-gain-v293.json`
先于运行提交。使用 v292 已消耗的 6 条纯脚但偏靶球、3 条高质量
保护球，比较 0.8 与 1.0 增益；两档均负侧启用、横向速度上限
`±0.2 m/s`，每档独立父回放与摆脚回放。四卡 A6000 完成
9/9 球位、36/36 次正式物理回放，零收集失败。正式证据目录
`/code/rosclaw/rsi-approach-intermediate-gain-v293-audited/`，
汇总哈希
`sha256:c0f91155277256c7f5287ebfc7b6bf8bac74e8e074389072b2141e611f3c729f`。

| 九条开发球位 | 0.8 | 1.0 |
| --- | ---: | ---: |
| 高质量出球 | 3 | 6 |
| 纯脚触球 | **9** | 8 |
| 平均物理回报 | **3.294** | 2.813 |
| 三条原好球保持 | 3 | 3 |
| 新增整段横向出界 | — | 0 |

在六条偏靶球中，seed 75 lane 4、77 lane 4、80 lane 4
转为高质量。seed 74 lane 2 的 1.0 档首次接触仍出现碰撞体
`[0, 5]`（脚和右膝），而 0.8 档仅 `[0]`；该球与另两条
偏靶球没有救回。预注册门要求所有九条球保持纯脚触球，故
`clean_foot_retained=false`、`development_gate_passed=false`。
更低的平均回报也反对用高质量球净增来单独宣称整体进步。

本轮启动前有三次**未纳入正式统计**的预检/审计故障，均在新证据
目录保留：首次 `/code/rosclaw/rsi-approach-intermediate-gain-v293/`
使用不含 Isaac Lab 的 Python；第二次
`/code/rosclaw/rsi-approach-intermediate-gain-v293-qualified/`
给错 SONIC 模型目录层级；第三次
`/code/rosclaw/rsi-approach-intermediate-gain-v293-validated/`
的两层独立审计白名单尚未接纳 1.0。修复审计白名单、测试通过并
提交后，才在新目录启动正式批次。第三次已有物理轨迹，但因审计
失败，**未复用或计入**正式证据。

下一步不能继续凭这九条开发球位微调增益。需要利用触球前的
本体/球几何序列，判断膝部接近风险，再测试能否在不牺牲脚触球的
条件下选择进攻性站位；只有开发门过关，才在未见新球位一次性考试。
此结果不是进球率、多人传接射扑率，也未达到连续比赛宣传片门槛。
