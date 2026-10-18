# 跨状态基线与已有滚离球能力：不重复发现旧策略

## 实际完成

contexts2–7的None/option4共12条历史密集回放已完成，36000子步，
冻结源7afa48d5。按原2ms力矩重放并核对300帧状态、执行器力、
骨盆高度及六通道接触，补记所有机器人与球大于1N接触。无NN调用、
新控制或权重更新；12端点仍只有6历史课程，新脚面实验不受影响。

外部`rsi-twelve-context-baseline-contact-replay-20261018/complete.json`：
`f58f137600577783488c176299ea30148aecb3e76ba34cacc1747f947d6ab24f`。
`rsi-twelve-context-baseline-contact-metrics-20261018.json`：
`a490aca9c36a4e1f3c55e0dcb23cdfae7313a03e9a0028461302c7602bd4254d`。

| 课程 | None任务 | option4任务 | option4首足/首非足子步 | option4机器人向下冲量 Ns |
| --- | --- | --- | --- | ---: |
| context2 | 通过 | 不安全 | 1666 / 1904 | 80.61491 |
| context3 | 通过 | 通过 | 1905 / 无 | .09628 |
| context4 | 不安全 | 不安全 | 980 / 1360 | 13.76112 |
| context5 | 通过 | 球路失败 | 972 / 无 | 7.96034 |
| context6 | 无足触球 | 晚触球删失 | 2716 / 无 | 36.33990 |
| context7 | 无足触球 | 晚触球删失 | 2728 / 无 | 3.11587 |

None-context4首足910、首非足921，其余None未见非足碰球。
这是阈值球接触描述，不是所有碰撞/自然动作证明。更晚触球、脚相位、
姿态和闭环轨迹已不同，不能单凭冲量推断因果或只保留成功子集。

新增`rosclaw-surface-thirtysix-contact-metrics-20261018.timer`已排队。
新6体及独立审计、summary、密集诊断均实际正常终态后，整合旧
4+3+5+12及旧/新脚面6+6共36端点。这仍只有12消耗公开课程，不是
Fresh；只读整合不增加积分/训练，现在排队不等于已经完成。

## 旧catalog2同C7成功，不能当新学习也不该遗忘

重新认证完整44体旧研究及独立审计，严格确认
`zero_control-2026101801-context-7`的candidate等于旧catalog[2]。
同C7第227帧触球，前向2.347285m、侧偏.510553m、骨盆最低
.696894m，原任务通过；catalog[4]第272帧触球窗口不足。
该零头的3000次均值、latent、请求与实际附加力矩全部严格0，
无随机采样、模型更新，不能把此旧成功写作PPO训练成果。

同组77、同planning context、初始canonical q/v逐位相同；世界、
Foundation、late actor、step model和父motor身份都相同。
但native源码hash不同，catalog2与4也有多项参数差异，故不是
同源码单一速度消融，不能证明只提高cap就必定有效。
该额外C7专家尚无本轮完整全身密集接触审查和附近扰动验证，
不直接进入教师/学生准入，不挑掉此前三个学习种子的失败结论。

发现的是已有较快接近且准确对齐的研究起点。不能因为一直使用
catalog4就说C7全系统不可达；后继优先复用catalog2做滚离球对照，
仍须保持集与多扰动。未来网络需要学习适用域，不能全局套补丁。

认证脚本`rsi_recover_archived_receding_expert.py`，冻结6e51c1c6。
外部`rsi-archived-catalog2-context7-recovery-20261018.json`：
`47c2d5b6df82a332bf29c30cf43924a26f3d074e1be46b2d89db831f408f5670`。
源receipt `8391d576d2dbbe801293439955e925bd5a4b13c8cea70e9037e0b3f83866acd0`。
此次新物理/NN/优化均0，teacher_qualified及training_authorized为false。

## 软件边界

范围回归40项、旧专家零控制检查6项通过，新源码Ruff/format通过。
全仓Ruff check及compileall src/tests通过。全仓format check失败：
68个旧文件需格式化；与冻结7afa48d5逐项对比，68个路径完全相同。
未批量改无关旧文件，不把全仓格式检查说成通过，更不把测试算技能。
