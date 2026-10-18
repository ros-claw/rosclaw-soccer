# 私有生成绑定入口：四体原生非干扰协议

目的仅是提高未来训练的数据准备吞吐，不是训练新球技，不替代当前
44体神经学习研究。之前真实大模型基准两seed局部准备提速4.38–4.42倍；
尚不能据此直接接入所有采集器或声称端到端倍速。

## 显式入口与不变量

`run_native_first_touch_episode`和`qualified_episode`新增
`generated_seed_episode_binding=False`。默认关闭；原preview/bind语句
原封不动保留在else。默认路径规范化AST仍等于原冻结hash
`65553cdb7c835e95fd63a2a20d987ebdb9237aaae1d7c17e4e4a07915cab88ce`。

打开时必须是捕获Foundation的非混合SIM development，精确私有factory，
reference decoder，完整原step_model输入；拒绝其他factory、伪bool、
body-response和cached proposal envelope。输入仍由原loader完整还原、
验证，只把后续重复preview+bind替换为私有seed生成＋绑定，并再次比较
生成的整个模型与原输入完全相等。原完整constructor和外部document
bind验证都不删减；承诺记录新的入口选择和原seed，CPU仍走原独立
reference compiler及全物理/Foundation/encoder复核。

本轮回归65项通过，变动两个生产模块mypy通过、Ruff/F821/F822通过。
其中包括5项非法入口提前拒绝和原默认物理AST不变；不是65场物理通过。

## 固定四次物理执行

同一新冻结源码、同mesh世界df743dfc…、同300帧/3000子步、None任务
控制、冻结完整旧mean、无新小脑或优化。原新世界group4为成功，group8
为球路失败。只选这两个原始draw以分别检查成功和失败的非干扰，顺序：

1. group4原入口。
2. group4新入口。
3. group8新入口。
4. group8原入口。

总预算4新native body＋4独立CPU复试，12000＋12000子步，1200次
Foundation与1200帧encoder复算；零优化、零新candidate。每个body
执行前先持久化intent并收费，普通失败保留；不因失败换seed/球位。

要求每个新轨迹的**全部原数组**与对应已审计原始新世界控制bit-exact，
同源码旧/新入口之间也bit-exact（dtype/shape/bytes）。原M0成功/失败
标签逐字段保留，不能把保留原失败当新成功。完整四份独立审计、见证
与encoder比较后才写complete。中途不一致或合同错误则记录fatal并
停止，不自动重跑或热改源；它只拒绝此优化，不改变当前44体研究。

## 执行与结论边界

runner：`scripts/rsi_qualify_generated_binding_native.py`。
静态preflight应零模型分配，展开源先过F821/F822；全部源/config固定
冻结HEAD后才分配。入场≥56GiB MemAvailable，最多两个本项目大模型
worker；另一worker为原44体考试，不停止或修改。硬时间预算一小时、
MemoryHigh8G/Max12G/Swap4G、CPU150%、Restart=no。

记录每例完整qualified_episode耗时，但首例冷热缓存、不同顺序和少量
样本限制结论，不将两个局部准备倍速直接写成整个pipeline倍速。
四体成功终态后还需独立终态回读；只有该同世界/无新小脑路径取得
资格，不能自动外推到其它policy family、CERE组合或真机。

所有晋升/硬件/教师标志保持false。证据外置，代码直接提交main。

## 实际启动记录（尚无四体终态）

冻结源3920e4dafbd5591c6b39d84f980594a82c2d3dc3，preflight实际通过，
未分配模型/世界。13:53:24用户级服务
`rosclaw-generated-binding-native-parity-20261018.service`已启动，
PID1929546，入口pins/精确干净HEAD/旧worker槽位守卫及F821/F822通过。
13:53:32验证归属历史报告、13:53:48进入独立reference完整分配。

输出`rsi-generated-binding-native-parity-20261018`，独立配置
`rsi-generated-binding-native-parity-config-20261018.json`，固定deadline
1792306370。入场MemAvailable约58GiB；原主研究PID1760082仍active，
本项目合计两个大模型worker。只读watcher配置纳入这两个明确单元，
它不执行训练、自动重试或科研继任决策；active Goal继续负责后续。

原生/小脑/当前课表/保持/传递诊断等关联回归310 passed（16.13秒），
不是全仓测试或310个物理案例。现在不能填写原生非干扰通过或端到端
提速倍数，必须等四个正式收据和独立终态回读。

## 14:26实际终态与独立读回

四个body全部完成并经过原CPU/Foundation/encoder复核，原数组每例28个、
总112次与原同世界控制的dtype/shape/bytes比较均相等。group4原/新均
成功，group8原/新均球路失败，未继承或改写失败为成功。

complete封印`sha256:4e28cfb8e47ceada5916968d3cdb5384af9c8fa277e12a90402d0f9e2e74a29b`。
独立审计冻结c4bf941b，使用既有reader扩展的独立binding-four profile，
14:25:55启动、14:26:39成功终态。完整账本、原生/独立复核/Foundation/
encoder见证字节、完整共享payload逻辑封印、真实原/新入口tag和原始
全部数组/标签再次核验。新物理和新优化均0。不是重新复试第二遍物理。

独立读回`rsi-generated-binding-native-independent-readback-20261018.json`，
封印`sha256:4ce439a7ecf8ad06edbd4c5de255ffacc8f9c0e11efbda5c73d04da86a9cecb4`。
账本tip`sha256:6eed5dc2900265c29bbb13270d1af2d7eaa7bde678137f383cf609c692ca73a2`。
资格范围仅为当前df743dfc…世界、raw4/8、None控制、无新小脑；其他
control profile/世界/神经组合仍未自动取得资格。

| 原生draw | 旧qualified_episode秒 | 新qualified_episode秒 | 两例局部比值 |
| --- | ---: | ---: | ---: |
| 4 | 224.647701 | 174.060911 | 1.290627 |
| 8 | 221.456079 | 171.603390 | 1.290511 |

这是各qualified_episode内部native＋独立CPU＋Foundation路径的实测
耗时（约少22.5%），不含整次完整初始化、外层encoder及全部产物检查；
样本少且存在缓存/机器负载影响。不能写成全pipeline稳定倍速，更不能
把之前4.4倍准备阶段倍速替换成整体倍速。没有足球能力净增益。

旧生产者没有把prepare的完整PINS/HEADS映射存入声明。本次依冻结的
原准备器及原有manifest重新派生并验证，读回明确标记这一限制，不能
宣称这些映射已由旧生产者保存。main后续版本将直接持久化这两项并
要求严格比对；不改旧证据、不重跑四体、也不热改正在运行的catalog。
