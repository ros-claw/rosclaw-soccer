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

## 实际启动与接续信息

冻结源码`4f68c7e502ed1c60b85e94c14670978bb653d44a`，零模型/零世界分配
静态preflight通过，全部展开源F821/F822与启动pins守卫通过后，服务
`rosclaw-generated-binding-benchmark-20261018.service`于13:18:36启动，
PID1875900。输出外置于`rsi-generated-seed-binding-benchmark-20261018`。

运行上限45分钟、MemoryHigh6GiB/Max10GiB、CPU150%、Restart=no；入场
可用内存约60GiB。与PID1760082的学习活动合计两个大模型worker，不
修改原学习源码或抢占无关GPU任务。该服务无新物理或学习更新预算。
13:21仍在完整原构造器初始化，尚无完整实际模型测速或非干扰结果。

## 13:32真实完整模型终态：局部准备路径获得可复现数值一致提速

服务已正常终止（用户级LoadState=loaded、MainPID=0、inactive、success）。
原完整JSON载入11.468590秒、原完整资格构造621.651437秒，未省略。

| seed | 执行顺序 | 原生成＋bind秒 | 新生成＋bind秒 | 局部倍速 |
| --- | --- | ---: | ---: | ---: |
| 30300 | 原→新 | 49.112513 | 11.116038 | 4.418167 |
| 30301 | 新→原 | 49.043778 | 11.205172 | 4.376888 |

两例各300帧、共600帧完整普通JSON文档/原hash、动作、隐状态和全部
采样transition逐项相同，新旧episode状态不共享；权重仍共享，身体输入
是同一历史轨迹，不能据此宣称原生物理非干扰或独立decoder实现复试。
实际新物理、优化更新均0，当前44体研究未启用此接口。

完整benchmark封印：
`sha256:593a3ca04531f97adf2e97ddb94d9d2b92ab4583c2e419212b21ddc951b8850e`。

使用另冻结的cc9f53cd脚本独立回读，验证2856份原source/model/body/world/
stage文件字节、全部逻辑封印、双顺序、计时绑定及比值；未再次分配完整
模型，也未重新执行decoder或物理，不能重复计为新600帧数值复试。
回读封印`sha256:ca3c9a0dadfc7107725fd6a567d1b664fc39c17e73b425c468940bb5d332df68`，
外置文件`rsi-generated-seed-binding-independent-readback-20261018.json`。
回读专项7项测试、Ruff、F821/F822通过。

发现：外部document bind中的完整资格重复检查确实是局部准备开销；
私有生成路径在保留最初完整资格及外部bind合同情况下可节省约38秒/例。
排除：不是以小模型替代真实模型，也不是略过原完整初始化获得倍速。
未证明：原生动力学非干扰、端到端采样倍速、神经学习或足球能力净增益。
下一步若接入，必须另声明seed-only原生入口，并在同世界同初态下与原
入口做完整物理/独立复核/数组bit-exact对照；任何差异均推翻接入资格。
不得为加速而改变正在进行的44体协议或跳过必要证据。
