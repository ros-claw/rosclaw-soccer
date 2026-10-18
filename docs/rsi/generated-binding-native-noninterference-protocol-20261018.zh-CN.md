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
