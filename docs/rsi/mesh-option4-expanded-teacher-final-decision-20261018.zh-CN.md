# 扩展教师screen终态：拒绝将option4用作全局教师

18个新身体于16:54:50全部完成，用户级worker16:54:52正常退出。
冻结5329aa6d的独立终态reader于16:55:24自动接续，现已成功退出。
不是以运行中Result=success或部分positive提前终结。

## 正式结果

复用原6端点，加18新端点，共12个独特开发课程、24配对端点。
其中8个课程为四类各两次固定扰动，另4个原始课程并非四类均衡，
部分源场景相关；不称24独立场景，也不声称12个IID随机抽样。

| 指标 | 原None对照 | option4 |
| --- | ---: | ---: |
| 任务通过 | 6/12 | 5/12 |
| 原失败救回 | — | 2 |
| 原成功回退 | — | 3 |
| 相对对照新增不安全 | — | 3 |
| 新优化器更新 | — | 0 |

救回raw8/56；回退raw4、context2、context5；新增不安全为raw4、
context0、context2。context4原本已不安全，两端点都失败，不重复
计为新增不安全。context6/7新出现足触，但271/272帧之后观察不够，
仍是RIGHT_CENSORED，不是两个任务救回。

**决定：拒绝全局option4教师，拒绝无条件用于学生蒸馏/晋级。**
新增执行18×3000=**54000原生子步及54000独立CPU子步**，
5400次原生Foundation调用及对应独立复算、270次额外encoder planner调用。
重复复核不计新物理/训练。没有private Fresh、52场开发考试、神经净增益
或球队比赛资格。现有全局Goal保持active，不把此拒绝当项目结束。

## 独立证据

- 输出：`rsi-mesh-option4-teacher-screen-20261018/complete.json`
- complete seal：`sha256:45a893b3bcfe9921c3d9d1172ac4335f4b39f66dcf007e1b70b00ab48304aa71`
- 独立读回：`rsi-mesh-option4-teacher-independent-readback-20261018.json`
- readback seal：`sha256:e42f0d2f7ce1c91fde7e649e3d50907a645ee3448c1ecfe632c4b32f813aa863`
- ledger tip：`sha256:a11df2e42ac2c6354034f81c8041f15762ed9ed89356e09ce2e5a6086cd28c48`
- reader字节：`sha256:fb99b1b977d20ce854a0b84e3bdc4a8885a63d5dfb8d366e5c51e2ef9ee3afbb`

读回已验证全部账本、新收据/见证、共享数值payload封印、原始初态和
sampling seed、8个完整planning context、encoder捕获、CPU/Foundation
资格，并以先前两个完整readback认证6个复用端点，重算12对结果。
本生产者的prepared_pins/source_heads原始快照确已保存且逐项相等，
不是事后补齐旧生产者缺口；旧16体catalog未保存快照的限制仍保留。

## 为什么不是再盲目换个catalog选项或多训PPO

高频重放已将多条失败定位到长时间足压球，见
`contact-exit-seven-course-findings-20261018.zh-CN.md`。raw4则在任何
足触之前已非足部碰球，单改触球后恢复不可能消除这一已发生的误触。
远处滚离球又存在纵向速度/横向对齐/脚相位的不同瓶颈，见
`receding-ball-approach-diagnosis-20261018.zh-CN.md`。

因此当前接续为已运行的4体“取消残留摆腿”因果消融，不改样本或预算。
它只检验接触退出，不声称修复所有接近与触球几何。接下来如需改变
接触点，则复用已完成实际G1点Jacobian与真实mesh表面IK原型，另外
声明完整闭环对照，不能用静态10/10接近目标替代真实踢球成功。

更高层接球、传球、射门、团队协作仍未由本实验完成；不将forward-ball
first-touch任务结果直接填为控球/传球成功率或successor readiness。
