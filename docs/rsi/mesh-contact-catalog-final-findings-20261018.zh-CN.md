# 固定catalog完成：两种已知难例存在安全可达解

## 完整执行与独立结果

生产源7b112e83的16体研究于15:44正常结束。冻结c4bf941b的独立审计
于15:44:51自动接续，现已实际成功终态、PID0。不是部分前缀或只读日志。

完成封印：`sha256:8e3a49180a53d10b1b0b4ead462932e7b54f8f9497bf2e9a4b2b0ab99482a654`。
独立文件 `rsi-mesh-contact-catalog-independent-readback-20261018.json`，封印：
`sha256:b3ea65c2f091d51e99a4d9599fde1618e5ae01ab8f840ef9788c1ec4eab6b5ef`。
ledger tip：`sha256:e6975143a564a96d1e0ee741b79004246babb9a6d938e513560960ba05c732d4`。

| 固定option | raw8 | raw56 |
|---|---|---|
| 0 None | FAIL | FAIL |
| 1 默认参数/legacy gate | FAIL | FAIL |
| 2 默认参数/bounded gate | FAIL | FAIL |
| 3 bounded参数组 | FAIL | FAIL |
| 4 bounded参数组 | PASS | PASS |
| 5 默认参数/后触球导航 | FAIL | FAIL |
| 6 legacy gate/参数组 | PASS | PASS |
| 7 bounded参数组 | PASS | PASS |

16体全部安全；FAIL均为真实球路未达原门槛，不是基础设施故障。
6/16是这个有限选项网格的通过数，**不是泛化成功率**。这里只有两个
不同的已消费初态，不是16场不同比赛。两个None控制的全部原始数组
与此前同世界控制逐字节一致；没有为了新结果改动原来失败的初态。

两个课程都在同一“较远x、负向来球”类别：

- raw8：(2.6221276, −0.0352590, −0.6586988)，source seed20261060/lane8；
- raw56：(2.7577566, −0.0977729, −0.6729384)，source seed20261063/lane14。

因此不能把这些结果推广到四种类别、追赶滚离球、高球、守门或射门死角。

## 真实工作量与证据上限

16体原生48000子步+独立CPU48000子步，4800次Foundation/encoder
重算、240次独立planner ONNX调用。固定mean、原始父motor与世界DF，
新optimizer更新为0。终态审计新增物理/优化也为0。

完整重验包括原始文件字节与共享载荷逻辑封印、整条ledger、先于body
落盘的intent、世界/课程/采样seed/option绑定、raw primitive结果与
既有独立CPU/Foundation证明，以及encoder完整消费时序。

旧生产者未持久保存prepared_pins/source_heads两图的原副本。审计从
当时准确的preparation源码/资产manifest重新推导，并验证其可读字节；
报告明确 `original_prepared_pins_persisted_by_producer=false`、
`prepared_pins_rederived_from_exact_original_preparation=true`。不把它
重写成原时点快照已保存，也不靠重跑物理隐藏这个历史缺口。
新18体生产者已经保存两图，另一个审计将严格比较，不沿用该缺口。

## 发现、排除与下一方向

发现：现有组合控制能够在同一真实身体/球接触世界安全改变两例失败。
不能再把这些状态简单称为该身体/接口全局不可达。

没有证明：优化过的12D接口一定无能、B导航或C接触某一个模块独自
造成收益、固定option能够跨状态泛化，或新神经小脑已经学会。
各option同时改了多项参数和gate，仍需要同其余条件的消融才能分配
因果贡献。详见 `mesh-catalog-control-attribution-20261018.zh-CN.md`。

已实际接续而非只提建议：18体option4跨状态/扰动/保持筛查PID2077170，
冻结8d748da6；15:47:12完整bundle就绪，开始control-context-0。
不因完整catalog发现option6/7也通过而换掉当前候选。完整新矩阵才决定
是否扩大教师、隔离适用域或回到接触机制，不重启已失败的同族PPO。

本阶段不是三seed学习的成绩；原学习5/8、6/8、4/8、零救回的拒绝
结论不变。没有教师自动资格、Fresh、策略/硬件晋升或可宣传球队结果。
