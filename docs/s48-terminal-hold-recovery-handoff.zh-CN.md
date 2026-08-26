# S48：G1 起身末端保持、热交接与多卡物理复测

## 结论

S47 的 0/4 最终稳定问题已在组件物理考试中突破：固定倒地姿态 4/4、局部扰动
倒地姿态 24/24，合计 28/28 完成“神经起身专家—循环 locomotion 小脑”交接，
并在 20 秒考试末尾连续稳定 6.14–9.10 秒。当前证据上限是
`PASS_SIM_COMPONENT_LOCAL_PERTURBATION_28_OF_28`，不等同于完整扑救落地分布，
更不代表真实 G1 授权。

这次没有通过盲目增加 PPO 训练量解决问题。根因是技能终止语义和控制器交接错误：
旧实现刚越过较低的高度/直立阈值就切断起身专家，而此时身体仍滞后于参考动作且
带有动量；随后又让冷启动的循环 locomotion 网络立即接管。继续训练只会让策略
学习补偿这个错误接口。

## 修复后的闭环

1. 起身动作到达最后一帧后不立即终止，神经专家继续围绕其站立末帧闭环控制；
2. 只有骨盆高度不低于 0.72 m、直立投影不低于 0.95、根线速度不高于
   0.25 m/s、根角速度不高于 0.50 rad/s，且双足同时支撑时才累计稳定时间；
3. 上述状态必须连续保持 1 秒，中途任一条件失效都会清零；
4. locomotion LSTM 从首次进入稳定包络时清空旧记忆，随后只读同一因果本体状态，
   但在稳定保持完成前没有动作权限；
5. 专家与 locomotion 的关节目标、KP、KD 用 1 秒连续混合完成权限交接；
6. 晋升考试要求交接完成后，考试末尾连续稳定至少 2 秒，不能用瞬时站立或
   auto-reset 后的默认站姿计分。

这套“终端闭环保持—连续完成判据—下游策略预热—渐进权限转移”并不限定于足球，
可复用于抓取后复位、碰撞脱困、移动底盘重新定位和其他具身技能组合。

## 实验结果

| 设备 | 世界数 | 初始条件 | 最终稳定 | 末尾连续稳定范围 |
|---|---:|---|---:|---:|
| A6000 GPU0 | 4 | 精确源倒地姿态 | 4/4 | 7.74–8.62 s |
| A6000 GPU1 | 8 | ±0.05 局部姿态/速度扰动 | 8/8 | 7.58–8.64 s |
| A6000 GPU2 | 4 | ±0.05 局部姿态/速度扰动 | 4/4 | 7.66–8.98 s |
| A6000 GPU3 | 4 | ±0.05 局部姿态/速度扰动 | 4/4 | 6.14–7.80 s |
| GPU0–GPU3 设备自描述复测 | 8 | 每卡 2 个局部扰动世界 | 8/8 | 7.64–9.10 s |
| 合计 | 28 | 固定与局部扰动 | **28/28** | **6.14–9.10 s** |

原始证据：

- `/code/rosclaw/rosclaw_football/evidence/s48-terminal-hold-handoff-v1/physics-probe-fixed-seed-47201.json`
- `/code/rosclaw/rosclaw_football/evidence/s48-terminal-hold-handoff-v1/physics-probe-perturbed-seed-47211.json`
- `/code/rosclaw/rosclaw_football/evidence/s48-terminal-hold-handoff-v1/physics-probe-perturbed-gpu2-seed-47212.json`
- `/code/rosclaw/rosclaw_football/evidence/s48-terminal-hold-handoff-v1/physics-probe-perturbed-gpu3-seed-47213.json`
- `/code/rosclaw/rosclaw_football/evidence/s48-terminal-hold-handoff-v1/device-bound-gpu0-seed-47220.json`
- `/code/rosclaw/rosclaw_football/evidence/s48-terminal-hold-handoff-v1/device-bound-gpu1-seed-47221.json`
- `/code/rosclaw/rosclaw_football/evidence/s48-terminal-hold-handoff-v1/device-bound-gpu2-seed-47222.json`
- `/code/rosclaw/rosclaw_football/evidence/s48-terminal-hold-handoff-v1/device-bound-gpu3-seed-47223.json`

局部扰动包括根姿态、29 关节位置、根线/角速度和关节速度的小扰动；它能检查专家
对源轨迹邻域的稳定性，但不能替代从真实扑救轨迹抽取的左右侧卧、俯卧、手膝支撑
和带横向冲量的恢复初态。

## 多卡缺陷修复

第一次在 GPU1 运行扰动考试时，Torch 张量和 MJWarp 数据已经创建在 `cuda:1`，
但后续 broadphase kernel 仍通过 Warp 全局默认设备发到 `cuda:0`，导致
`illegal memory access`。根因是构造期 `ScopedDevice` 没有约束后续 forward/step。

现在每个一进程一设备的 MJWarp 环境都会显式 `set_device`，并立刻验证 Warp 实际
设备与请求设备一致，不一致则 fail-closed。四份 v3 证据分别自描述
`cuda:0/1/2/3`，每卡 2/2 通过完整 20 秒考试。这证明四张卡可以各自执行该物理
考试，但本阶段没有声称完成同步四卡 PPO。

## 为什么本阶段没有训练新的残差网络

修复终止和交接语义后，冻结的专家与 locomotion 已经达到 28/28。此时立即训练
残差会增加参数漂移、破坏原技能，并掩盖接口修复带来的因果收益。下一阶段只有在
“扑救—落地”真实状态分布中出现当前专家无法覆盖的失败簇时，才训练有界的
`recovery-transition residual`，并冻结起身专家、扑救策略与 locomotion 小脑。

计划中的训练数据应来自失败路由器保存的物理快照，而不是直接把最低 mocap 帧
硬塞进接触场。每个快照需要绑定扑救技能、场景、身体、接触、速度和失败原因哈希，
课程按源姿态邻域、左右侧卧、带冲量落地、完整扑救链路逐级扩展。

## 验证与边界

- 相关单元测试、Ruff、mypy：通过；
- Torch/ONNXRuntime 数值等价：保持通过；
- GPU0–GPU3 MuJoCo-Warp 真实物理步进：通过；
- 组件晋升：`PASS_SIM_COMPONENT_LOCAL_PERTURBATION_28_OF_28`；
- 完整“射门—扑救—落地—起身—继续守门”晋升：尚未申请；
- 四卡同步 PPO：未运行，也不需要用来修补本轮接口错误；
- 真实机器人命令：0，所有能力均为 `SIM_ONLY`。
