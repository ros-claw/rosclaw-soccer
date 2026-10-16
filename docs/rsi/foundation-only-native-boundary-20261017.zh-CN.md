# 纯 Foundation 消融入口的防误标加固

本轮没有新增训练或球技成绩。核对后续残差消融入口时发现：历史 CLI
拒绝 `--foundation-only` 与神经/电机/step 模型混用，但原生函数由
研究 runner 直接调用时未复用这一前置限制。该函数的
`foundation_only` 会关闭任务空间摆腿门，却不能自行阻止传入的
step 模型继续生成残差，报告的执行 profile 因而可能被误读。

新增原生分配前检查：Foundation-only 必须是布尔值；开启时拒绝
学习模型、body guidance、所有七类策略工厂、缓存模型 envelope
及非 reference 残差解码器。正确的纯 Foundation 模式和原有非纯
Foundation 模式仍可到达分配入口。拒绝发生在导入/分配仿真模型前，
不是在跑完后按失败标签补救。

18 项新增 fixture 测试覆盖模型、工厂、解码器/缓存、标志类型和
两种合法模式的 mock 分配边界。与原生边界、走/跑、参考/编码器、
评分、Campaign、组合与配对研究相关的 187 项测试通过；Ruff、格式
与 compileall 通过。原默认物理控制体的冻结 AST 哈希仍为
`b49c6c901799b82c56e3a0bb663ffd7e30720f6ac5329e85b675ce2d99c012c2`，
未修改期望值来迁就代码。

类型检查分开记录：普通 mypy 在导入的
`growth/runtime_finish_plan_actor.py:663` 与
`training/goalkeeper_whole_body_reach.py:407` 报出两项错误；在未改动的
冻结源 `9a43ba3` 上复现了相同错误。目标文件采用
`--follow-imports=silent` 的检查通过，但不能因此宣称全项目 mypy
通过。本次不修改这两个无关模块。

当前活跃与排队研究使用独立冻结源码，均不受这次主分支改动影响。
本轮不修改碰撞体、球尺寸、导航、关节目标、力矩、奖励或任务时限，
也不重新标注旧结果。现有 bounded 接口仍要求 recurrent sampler，
不能把这个标志直接当作已完成的 bounded 导航/残差零作用配对实验；
若要开展确定性均值或无残差对照，必须另行声明对应执行与复核合同。
