# 触球后残留摆腿的四例因果消融

## 问题与证据上限

接触重放见 `contact-replay-foot-pinning-findings-20261018.zh-CN.md`。
失败context0首次击球方向正确，随后足部连续接触191个2ms子步，
向下冲量59.17Ns，球在小腿碰球前已反向。不能只修改瞄准点，
也不能据此认定20帧残留摆腿是唯一原因。

本试验先隔离最小机制：首次观测到原六通道任一力超过1N后，
从**下一个控制tick**开始将taskspace摆腿增量严格置零。Foundation、
原循环motor、导航控制律和参数、初始课程、种子、动力学与限制不改。
闭环观测变化后父模型输出可能随之变化，不能宣称整条命令相同。
这不是新恢复控制器，也不是强化学习或小脑训练。

## 固定四例，不按本次结果选样

| 配对 | 原始端点 | 原结果 | 新身体 |
| --- | --- | --- | --- |
| raw8 | 16体catalog receipt0008 | 成功 | 取消残留摆腿 |
| raw56 | 16体catalog receipt0009 | 成功 | 取消残留摆腿 |
| context0 | 18体screen receipt0001 | 不安全失败 | 取消残留摆腿 |
| context1 | 18体screen receipt0003 | 成功 | 取消残留摆腿 |

全部为原catalog option4，复用四个已提交基线；新增4身体、原生12000
子步、独立CPU12000子步，4独立课程/8配对端点。不是8个独立课程，
不是G3完整鲁棒教师考试，不访问private Fresh。所有失败原样入账。

每例完整300帧。新旧trace所有数组的集合、dtype、shape及从初始到
首触帧（含）字节必须完全相同；不一致即证据违约，不能继续把后段差异
归因于此次消融。首触之后实际taskspace增量必须全零，独立CPU捕获
校验器验证这一点，终态reader再次从原始数组重算。

原始身体安全门和M0球路门不改。去掉脚压球却导致倒地、误触或成功
保持回退，依然失败。即使四例改善，也不能直接发教师/策略资格。

## 已实际启动的执行与复核

生产者冻结commit `2380e409`，独立终态reader冻结 `fd1712d6`；
均为运行外另外创建的不可变worktree，不热改旧18体研究。
四例的零模型分配预检、源/配置/历史收据字节校验已通过。
16:34:27用户级服务 `rosclaw-swing-release-ablation-pilot-20261018.service`
实际启动，PID2177253。16:34:53正在分配完整独立参考，尚无新身体结果。

原始配置、启动pin与所有原始输出位于外部evidence根：

- `rsi-swing-release-ablation-pilot-config-20261018.json`
- `rsi-swing-release-ablation-pilot-launch-pins-20261018.json`
- `rsi-swing-release-ablation-pilot-20261018/`
- `rsi-swing-release-ablation-independent-readback-20261018.json`（等待终态）

截止为18:15，服务RuntimeMax为2小时、Restart=no，单次启动标记禁止
科学失败静默重跑。启动前MemAvailable至少56GiB，本研究占第二个
完整模型worker槽，不启动第三个。观察器已切换为18体与此4体。

终态审计timer已启用：除四例complete外，还要原18体与16体catalog
各自独立终态审计完成、服务loaded/inactive/PID0/success才开始。
独立reader验证全账本、全部新收据/物理见证、源和输入pin、原始初态、
真实Foundation/encoder捕获、逐数组首触前缀及精确零增量；并通过两个
既有终态readback认证原四个基线，重新计算4对rescue/regression/new unsafe。
复核不新增物理、不新增梯度，不把重算计入训练量。

## 下一分支

若失败反例明显改善且原成功保持，再在更广固定课程中检验接触分离、
身体恢复及下一技能就绪；四例不足以恢复全局option4教师资格。
若失败仍在，则不能声称删20帧衰减解决问题，应据接触法向、
脚/球相对速度、对侧支撑与压力持续时间设计有界抬脚/随摆教师，
而不是盲扫20→10→5帧或无依据放宽RL的KL/安全门。

教师的物理净收益与保持成立后，才将完整相位数据用于神经学生训练。
当前没有学习突破，也没有新的宣传视频资格。
