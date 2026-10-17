# 规划入口故障复盘：默认参考观测，不强加模式探针

原pilot正常完成完整factory及独立参考compiler初始化，随后在
第一个option（原父代A、candidate=None）进入原生API时失败：
`run-mode probe requires an unmixed bounded SIM control profile`。
模式探针要求有界且不混合postcontact控制，运行器却对完整八方案
统一传入模式2。因此不能只移除父代A，或跳过postcontact方案。

旧进程已终止（failed、MainPID=0）。旧FATAL账本、声明、上下文
文件与完整共享payload保留，没有physical_trace、编译世界或
完成收据。该异常在原生API的分配入口前发生，实际新身体0，
已收费attempt1、completed0，不把初始化伪称物理或训练成功。

旧账本封印：
`sha256:6677b5925bda39eaa0a417db12362a50422205d1999b29b00e649f30bfbbc87a`。
旧tip：
`sha256:50e774a2f44dc1c79d316b20e0357736d3c1f4ef85bfffc054a9661e0e050a1b`。

## 修改

增加明确opt-in的default planner event capture：只记录真实ONNX
feed中的mode与原有reference hash，不向feed注入模式、参考高度、
新动作或新噪声。默认false时原导航契约不增加字段。
新扰动规划要求无模式probe，自动启用默认参考记录；允许原目录
全部控制形状，不放宽模式3的原禁混合边界。

使用原独立reference reconstruction及encoder消费检查来审核
捕获的默认mode2参考，仍不把观测元数据当作物理任务通过。
原CPU3000子步、Foundation300次重算及270个draw审核全部保留。

新测试逐个比较0、0.02、0.8、1.4m/s下带／不带记录的所有ONNX
feed key、dtype、数组及返回参考完全相同；原父代、有界、postcontact
形状均在mock分配边界前被接受；错误flag与混合probe拒绝。
原冻结AST检查只特化明确关闭的新增观测分支，不改冻结hash，
不删除控制算术。初步相关117项通过、Ruff与三模块定向mypy通过。
这是软件证据，不代替实际新扰动身体或新的球技成绩。

## 有限、显式恢复

不能重开旧FATAL或清零它。另建明确reviewed successor，认证旧
账本完整哈希链、旧错误、active与attempt1，并记入新账本。
总attempt上限从原9显式调整为10：其中旧分配前失败1，最多9条
新身体（8规划+1复试），**新物理预算没有增加**。
完整八上下文72条物理总额仍包含本pilot；另记分配前失败费用。
不隐藏变更、不自动重试其他失败或扩展参数／上下文。

新runner须在分配完整模型前逐个预检全部八个实际目录方案，
并认证默认参考捕获开关；通过后才可进入原物理执行与审核。
仍无教师、学生训练、Fresh、Champion、宣传或硬件授权。

## 恢复预检与启动

扩展相关十组测试349项通过，包含旧模式配对、完整原评分组合、
默认冻结AST、规划域和动作选择边界；Ruff通过。
恢复执行源冻结为`4951751f7825052fb64a4f6504e4373a1d8e2ced`。
新runner `rsi_public_physics_planning_pilot_repair_20261017.py`
在分配前读回并认证旧FATAL完整哈希链，将旧收费attempt1
显式导入新账本，保留旧active／error，不能冒充旧已完成身体。
全八个实际catalog方案均通过mock allocator边界检查；新上下文
及选中前的持久化因果检查保留。截止时间沿用原声明，没有延长。
最终完整runner通过F821/F822和无物理／全模型分配预检，
配置封印：
`sha256:a1a56421310cd4a48bf32de2a43cb858904234ac62191103246a9a4f69399ff8`。

服务`rosclaw-public-physics-planning-pilot-repair-20261017.service`
已启动；当前没有完成结果，不能把初始化或进程存活视为通过。
当前只读参考审核每场额外15次planner调用，同时核对其300帧
encoder消费；原模式探针专用的另15次reference审核不再冒用。
原完整CPU、Foundation与采样独立审核数量保持不变。

## 第二次分配前故障：预检模块表恢复导致C扩展重复导入

恢复服务在预检结束后、完整模型初始化前导入torch时SystemError，
最终SIGSEGV。旧新两份FATAL账本与声明均保留；第二次active=null、
completed0，无physical_trace或compiled_model。它不是物理失败
或学习失败，不能算一条成功或训练样本。

定位到运行器的`patch.dict(sys.modules, ...)`：预检调用原生API
时首次导入torch；退出mock上下文整体恢复模块表，删除了期间
导入的torch条目，但C扩展原生状态仍在，再次导入导致崩溃。
独立进程正常导入torch 2.13.0+cu130；没有重装或降级依赖。

第三份独立runner只暂时替换／恢复mujoco这一条模块，不回滚
期间其他导入。预检后检查torch与torch._C仍正确注册，再运行
同一完整main到分配边界，最终F821/F822及完整预检正常退出。
旧运行器与源码不修改，不重开两份旧FATAL。

第二份旧账本：
`sha256:65bbd27f94db07bd1067ebef1e18191cfd2c006787944ffed54ecf3ba229c5c5`。
第二份旧tip：
`sha256:14c82c423be42124907e7a9060c120bdd758c8dfc2d776251aac4fbf8b387f8d`。
第三运行器显式认证两份完整哈希链；收费起点2（原入口失败1、
本次import失败额外计1），总attempt上限11，仍最多9条新身体。
旧deadline、新上下文、目录、原评分与72条完整物理总预算不变。

新runner `rsi_public_physics_planning_pilot_import_repair_20261017.py`，
服务`rosclaw-public-physics-planning-pilot-import-repair-20261017.service`。
这些修复仍不是已完成的物理规划或神经学习成果。
