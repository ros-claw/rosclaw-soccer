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
generated binding入口，因此不采用该性能优化的未证实路径。自动接续
只接受前置单元真实loaded、MainPID=0、inactive、Result=success；若其
合同失败，必须先由Agent诊断，再按独立声明继续足球研究，不能盲目
启动下一物理任务。最多两个本项目大模型worker，入场≥56GiB可用内存；
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

## 已冻结并排队，不是已开始训练

冻结源7b112e83969da4a15581e9efb0d4b9ddd6f4cbe7。
实际preflight输出`MESH_CATALOG_16_JOBS_ALL_7_INTERFACES_ZERO_MODEL_ALLOCATIONS`，
两项固定计划测试及Ruff/F821/F822通过。输出外置
`rsi-mesh-contact-catalog-reachability-20261018`；配置
`rsi-mesh-contact-catalog-reachability-config-20261018.json`，固定deadline
1792314563（本地17:09:23），不因等待延长。

用户级`rosclaw-mesh-contact-catalog-20261018.timer`已启用。实际验证前置
PID1929546活跃时，ExecCondition退出1，尚无模型PID、输出root或启动
marker，不占第三个模型槽位。自动接续实际使用更严格的success门，
`rsi-mesh-contact-catalog-launch-pins-v2-20261018.json`；原v1声明保留，
没有任何物理尝试或预算变动。一次性launch-attempted marker在入口
守卫之前落盘，守卫/脚本失败后也不静默自动重跑。

这让有限的下一科研阶段在已声明条件满足后可继续，而不是只有一个
不断打印状态的监视器；但本阶段尚未采集，不可称已经学习或可达。
