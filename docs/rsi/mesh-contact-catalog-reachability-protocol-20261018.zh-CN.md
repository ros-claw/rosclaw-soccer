# 新世界困难球位：原有界控制catalog完整可达性对照

动机：本轮全时域小脑PPO seed1已使raw4原成功丢失，raw8仍失败。
不能只扩大KL或再堆相同训练。按1013实施建议回到G2/G3，先检验接近/
对齐/接触控制在当前物理世界是否有解。

## 明确范围

仍为同mesh世界df743dfc…、同M0/300帧/60帧后窗/骨盆0.65m/真实足触
门限/硬力矩限制、完整冻结Foundation/父motor/原seed，无新小脑和
优化。只研究原审计四例里两个失败raw8/56；两个独立course、16次执行，
不能称16个独立场景、完整教师或球队能力。

复用原完整8-option catalog，封印
`sha256:2241743df8d88616c96a29106ec3c2b439238be485784706ae03d753f9552261`。
0为None，1–7为已有有界接近/接触/恢复控制，参数不再调整。穷尽全部
16个group×option组合，按option0…7顺序各执行raw8再raw56。最先的
两个None必须与原新世界轨迹的所有原数组bit-exact，证明基线未换。
不根据本轮结果追加候选/改球位；旧已知不安全option也保留实测负例，
不因label失败而以基础设施异常重试。

## 资源与持久化

runner：`scripts/rsi_mesh_contact_catalog_reachability.py`。原准备器、
FirstTouchCampaign账本、qualified_episode完整native→独立CPU→
Foundation→planner/encoder证据链全部复用，不造新训练框架。
全部候选有界接口先静态预演；preflight零模型分配。

4体准备优化研究释放槽位后才能分配本研究；独立控制默认**关闭**新
generated binding入口，因此不依赖该性能优化的成功结果。不论该
性能研究成功或失败，都不因此阻断足球研究，但必须MainPID=0且单元
真实loaded/终态。最多两个本项目大模型worker，入场≥56GiB可用内存；
MemoryHigh8G/Max12G/Swap4G、CPU150%、RuntimeMax2小时、Restart=no。
固定绝对deadline在配置写入时声明，等待不自动延长。

每次先写不可变intent再收费，出错保留账本/输出。最多16个native body
与16个独立CPU复试（48000＋48000子步），4800次Foundation、4800帧
encoder/240次planner复算，零优化。所有通过、球路失败、无触球、合法
删失和不安全接触保留，schema失败才作为合同异常停止；无隐式重试。

## 决策与推翻条件

完整16体及独立终态回读后，逐球位输出每个option的触球/球路/安全，
不要只报一个最好结果。若有成功，仅获得这两个已消费状态的可达性
证据；接着必须多扰动、原成功保持和跨状态验证，才考虑教师及学生。
若八种已有控制均不能安全解决，推翻“旧catalog已足够”的假设，
研究接触条件化的控制权限、观测或更可靠先验，不无限重复同族PPO。

当前44体三种子考试及终态审计完整继续，不因本协议改变候选、考试或
预算；没有未来结果时不提前宣称全部seed失败。所有教师/晋升/硬件
授权仍false。视频留待真实能力改善，不用新场景渲染掩盖负结果。
