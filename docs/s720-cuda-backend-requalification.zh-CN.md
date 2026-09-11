# S720–S723：区分 GPU 状态工具故障与可用计算，重新验证足球物理后端

## 当前结论

四张 RTX A6000 的 CUDA 计算实际可用，不能因为 `nvidia-smi` 失败便认定 GPU 不可用。本轮没有更新、卸载或重载驱动，没有停止其他用户进程；此前S700、S705、S713和当前S715仍是原生CPU物理训练，不追溯改称GPU训练。

当前内核驱动为595.71.05，系统用户态库为595.91.07；NVML报不匹配。但PyTorch 2.13.0+cu130在四个显式CUDA设备上分别执行矩阵乘法并同步，结果与CPU逐项相同。记录时每卡约43–44GB可用。这仅证明四设备计算，不是四进程分布式训练，更不是足球后端合格。

## 安装与隔离

原CPU环境缺少`mujoco_warp`及`warp`。显式启用GPU专项测试后，首次结果为5 passed、5 failed（缺依赖），保留于 `/home/dell/rosclaw_soccer_evidence/s720-gpu-damping-tests.xml`，没有把缺依赖记为通过或略过。

参考[MuJoCo Warp官方安装说明](https://github.com/google-deepmind/mujoco_warp)，仅向外部隔离目录安装`mujoco-warp==3.13.0`和`warp-lang==1.17.0`；原MuJoCo 3.12.0及CPU训练环境不变。该MJWarp发行版声明MuJoCo最低版本为3.12.0。目录为：

- 原始发行包：`/home/dell/rosclaw_soccer_evidence/s720-gpu-backend-qualification-v1/packages`
- 修正后的独立包：`/home/dell/rosclaw_soccer_evidence/s722-mjwarp-corrected-damping-v1/packages`

修正版通过进程级`PYTHONPATH`优先加载，只复制并修改`mujoco_warp`，其余依赖从原始隔离目录读取。没有向共享虚拟环境安装或替换包。

## 最新发行包仍复现旧阻尼问题

原始3.13.0实测8 passed、2 failed：自由关节和球关节的第一个自由度阻尼为零、后续非零时，GPU返回全零阻尼力，CPU非零，最大差0.0924。禁用阻尼的测试通过。对应记录：

`/home/dell/rosclaw_soccer_evidence/s720-gpu-backend-qualification-v1/damping-tests.xml`

这是S245已发现问题在新发行包中的重新复现，而不是新的球技突破。代码检查显示多自由度循环继续复用首个阻尼系数；[上游实现](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/passive.py)可供参考，但可变main链接不替代下面的本地源码哈希。

新增版本对应补丁 `patches/mujoco-warp-3.13.0-per-dof-damping.patch`：逐自由度读取线性和多项式阻尼系数，并避免首自由度为零使后续阻尼全部跳过。保持禁用阻尼语义，不修改球的物理参数、奖励或接触体积。

- 原始`passive.py` SHA256：`3f36f15ef6f75fe04c0883a08c41e5f0c18ca8ef0f7adbc34e73954f9c3e777f`
- 修正文件 SHA256：`3ad100d2eccfe7e82aee478beadc5e590eacecd8d0b61538b57314250fd0da67`

对原副本的`git apply --check`及修正副本的`git apply --reverse --check`均通过。只允许对匹配的隔离源码副本应用该补丁；保留上游许可证，不自动全局修补依赖。

## 验证结果及尚未证明的内容

修正副本的同一GPU专项为 **10 passed**，8.09秒，无跳过，记录：

`/home/dell/rosclaw_soccer_evidence/s722-mjwarp-corrected-damping-v1/damping-tests.xml`

四卡各自执行三组正负速度的实际forward/被动力内核，含非均匀线性和非线性阻尼、首项为零，最大CPU/GPU力差`4.214344e-9`。四设备使用相同修正内核哈希。记录：

`/home/dell/rosclaw_soccer_evidence/s722-mjwarp-corrected-damping-v1/four-device-damping.json`

该检查没有推进仿真时间，不能称为四卡训练或整条动力学一致。

S723完成4秒G1触球对照：S717前四个CPU成功状态，在单GPU四个世界中用同一冻结SONIC和身体策略、同一球及关节/力矩边界实际步进，逐2ms记录接触与身体状态。4/4触球标签一致，前0.2秒最大q/v差分别为2.765062e-5、0.0011374。但完整回合球位置最大分量差0.08994m、关节角差0.028537rad，不能说轨迹完全一致，更不能用来认证0.1m射门精度。排除初始化/编译后的回合运行12.30秒，不是与CPU公平配对的加速比。

S724/S725进一步对两个失败/成功入口原型生成32个全新扰动状态：先保存CPU标签（23成功、9失败），再运行同样初始q/v的GPU对照，不剔除不一致样本。结果32个标签全部一致，两类分别为100%；同时，前0.2秒q/v最大差仍达0.002338/0.114022，不能改称严格轨迹复现。此范围内支持粗粒度触球筛选的进一步试验，不代表更广泛身体、来球或精度任务的后端资格。

S726检查另一条加速途径：完全不运行物理，只在S717的32条完整200帧实测身体记录上，比较CPU ONNX与既有`FrozenSonicG1Torch`/`BatchedSonicTracker`的独立历史和关节目标。全部目标最大差4.624495e-6rad，小于预先1e-4rad门槛；纯推理累计时间CPU 38.09秒、GPU 1.99秒，约19.13倍。该比值只针对这批记录的推理，不是训练加速比，也不是新球技。

S727保持原生CPU物理不变，只替换该推理后端，32个新扰动的23成功/9失败标签仍全部一致。前0.2秒q/v差8.265249e-7/1.394195e-5，完整回合最大球位置分量差0.020636m、关节角差0.007513rad，仍需逐任务检查接触敏感性。仿真回合16.50秒，未与同次CPU重跑做壁钟配对，不能直接当作训练加速比。

S728进一步检查正在优化精度的S713末代136/30网络，保留动态神经参考朝向输入和原生CPU物理：相同32个初始状态的速度、精度和安全标签均与父实验一致（精度仍只有1/32），虚拟平面横向误差最大变化2.696079e-5m，小于预先5mm门槛；前0.2秒q/v差2.852412e-7/1.154583e-5。完整评估约19.19秒。仅据此允许该推理加速器继续用于同课程开发，最终候选仍需原CPU ONNX独立复验。

S729已在这条已检查的CPU物理/CUDA推理路径上启动8代小规模神经参数进化：29个关节输出偏置与1个参考朝向头偏置，8候选×4个明确复用的开发状态，先守住4个状态的速度/接触/安全条件，再降低最差落点误差。输出为普通数值网络权重，后续必须复验导出权重，不能把实验中的批量候选偏置直接作为运行时补丁。它不是额外PPO梯度更新；原S715连续PPO保留为独立对照，未中途修改。

这些动作遵循rosclaw-simforge的隔离依赖、实测物理和CPU验收要求。当前证据上限仍为SIM_ONLY。
