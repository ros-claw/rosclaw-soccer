# 私有seed生成并绑定：局部采样准备优化协议

## 为什么做，以及不做什么

既有真实完整模型基准测得，私有seed预览约11.3秒，但外部完整文档bind
仍约43秒。当前原生采样还会恢复view并重复preview/bind；这些开销限制
后续增加物理经验的吞吐。不能以此假设每一项新API已经更快，也不能把
局部提速替代真实学习净收益。

当前44体三种子学习活动继续使用冻结源f44b9509；本优化不进入该活动，
不改变其预算、采样、奖励、动作、独立审核和晋升规则。

## 新API边界

`RecurrentSamplingEpisodeFactory.generate_and_bind_seed(seed=...)`只接收
uint32 seed，不接受调用方模型、policy、采样法、validator或权限对象。
factory分配仍使用原始完整构造器验证整模型，包括critic和历史父代。
新API从私有冻结字节生成完整普通JSON预览及原封印，随后从原数值原型
创建完全独立的episode历史。前后仍检查冻结字节、源码和数值原型。

现有`bind(policy)`仍执行原`validate_preview`；不把外部文档当成私有
文档，不按对象id或model_hash字符串缓存信任。两种路径复用同一个原有
状态克隆实现，避免重置逻辑分叉。返回的完整文档不是外部bind豁免令牌。

新接口尚未接入任何native collector。数值测试不能授予原生物理资格。

## 软件验证

必须覆盖两类原完整模型：imitation父模型和带critic的clipped模型。
比较原构造器与新接口的完整预览、300帧动作、隐状态、采样draw/logp，
检查新episode彼此独立、prototype不变、坏seed和篡改文档拒绝。

首次专项测试1 failed / 2 passed：imitation模型没有`critic_parameters`，
测试错误地在两种schema上访问该字段。300帧数值比较已经完成，但不能
把整组测试称为通过。修正为按实际schema篡改critic或actor参数后重跑；
旧日志保留。完整相关回归结果另记，不能用“已启动”代替通过。

## 真实完整模型基准预注册

入口：`scripts/rsi_benchmark_generated_episode_binding.py`。
完整模型仍为adbee3d6，canonical mean为f0fd83ec，原prototype为2c68fc1e。
复用已消费真实身体序列和其原编译世界，只读取关节limits，不做物理积分。
不加载Foundation，不增加新身体，不运行神经优化器。

固定seed30300先旧后新，30301先新后旧，无cProfile。旧路径为现有
`preview_from_seed + bind`，新路径为`generate_and_bind_seed`。计时同时
覆盖完整预览生成和episode绑定，不能将一边只计克隆、另一边计生成。
完整初始化耗时单独报告，不从计时中偷偷去掉必要初始化。

每个seed比较两条完整普通JSON及原封印、同一历史身体输入下的300帧
动作/隐状态/全部采样transition，且状态不共享。权重共享必须明确记录，
不能写成独立权重实现审核或反事实物理。输入、源码及whole-model hash
执行前后固定；至少56GiB可用内存入场，最多两个大模型worker并存。

不一致即拒绝优化，保存fatal，不自动重跑。即使局部更快且数值一致，
仍需下一份独立原生非干扰协议与完整物理复试后，才考虑采集器显式接入。
本协议不宣称端到端提速，更不宣称G1足球技能进步或宣传视频达标。

## 软件回归终态

修复测试的schema假设后，专项＋原factory＋原owned preview三组完整
运行结束：42 passed，842.56秒，exit0。新接口和benchmark脚本Ruff及
format通过；factory单模块mypy通过；benchmark展开源F821/F822通过。
这不是完整项目的全量测试，也不是完整实际模型的时间或物理资格结果。
真实完整模型基准仍需另行冻结运行并记录终态，当前不填写推测倍速。
