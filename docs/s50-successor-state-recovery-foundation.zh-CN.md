# S50：Successor-State Growth 与 Recovery Foundation 第一阶段实施报告

日期：2026-08-20
状态：第一阶段工程闭环完成；完整扑救恢复仍未晋升
安全边界：`SIM_ONLY`，真实机器人命令 0

## 1. 结论先行

本阶段没有继续给旧守门脚本追加“落地后等几秒”之类补丁，而是按
`rosclaw_soccer讨论4.md` 把问题重新定义成：一个技能只有在身体连续进入下一个技能
的可用状态后才算成功。

已经完成的突破是工程结构和证据闭环，而不是宣称守门员已经学会任意姿态起身：

1. ROSClaw Core 新增任务无关的后继状态、探索/晋升双安全剖面、失败条件 Dream、
   能力前沿课程和单焦点可塑性租约；这些能力不包含 G1、足球或守门术语。
2. Soccer 新增 `Absorb / Get-Up / Athlete` 三专家 Recovery Foundation 契约、四帧纯
   本体感软门控、29DoF 动作混合、受控跌倒安全剖面和 `GOALKEEPER_READY` 后继状态。
3. S49 的 50 个真实扑救后快照进入离线路由闭环。初版门控错误地将 6/31 个高动量
   状态提前交给 Get-Up/Athlete；根据失败证据校准后，高动量状态 31/31 进入 Absorb，
   站立状态 7/7 进入 Athlete，唯一 prone 状态进入 Get-Up。
4. 现有 R0 神经起身专家在四张 RTX A6000 上完成更强的 ±0.10 局部扰动物理复测，
   16/16 成功，末尾连续稳定 7.68–8.94 秒。
5. 这 16/16 不能抵消已有的真实扑救后状态 0/9。进一步检查发现 R0 的参考起始姿态
   是 supine，而 S49 稳定后的分布是 8 个 prone、1 个 left-side。真正缺口是多姿态
   Get-Up 与落地 Absorb，不是 R0 末端稳定性。

因此，本阶段没有申请完整链路晋升；四卡 16/16 被明确标记为
`COMPONENT_LOCAL_PERTURBATION_NOT_TRUE_POST_SAVE_RECOVERY`。

## 2. 新的系统闭环

完整链路的目标语义变为：

```text
动态传球
  -> 跑动一脚射门
  -> 飞身扑救
  -> Absorb 卸载冲量
  -> Get-Up 多姿态起身
  -> Athlete 恢复可移动守门姿态
  -> GOALKEEPER_READY 连续保持 1 秒
  -> 才允许把“扑救恢复”记为成功样本
```

三专家门控只接收四帧本体信息：重力投影、骨盆高度、根线/角速度、双足负载、非足
接触负载、平均关节速度和动作变化。部署接口没有 `stage`、参考动作相位、教师标签或
模拟器任务真值。三个专家输出相同的规范 29DoF 动作后才可连续加权混合。

`GOALKEEPER_READY` 不是“站起来一次”，而是以下九项同时连续满足 50 个 50Hz
控制步：骨盆高度、直立度、根线速度、根角速度、双足支撑、面向球场、处于守门区域、
手部准备误差、横向再加速能力。任一项越界都会把连续计时清零。

## 3. ROSClaw Core 实施

新增模块：

- `continual/successor_state.py`：内容寻址的 `SkillSuccessorState`、严格有序观测、连续
  保持计时和带后继价值的 Growth 目标。
- `continual/safety_profiles.py`：将允许接触式探索与严格晋升彻底分离；探索通过不产生
  晋升权限，头/颈接触、非有限值和越权动作在两类剖面中都 fail-closed。
- `continual/failure_curriculum.py`：失败条件 Dream、扰动分布、固定样本混合和优先训练
  30%–70% 成功率能力边界的课程调度器。
- `continual/plasticity_lease.py`：一次训练只允许一个 focal agent 的策略哈希发生变化；
  冻结队友漂移或优化步数超预算都会拒绝本轮证据。

这些合同与既有 Champion Registry、个体 Growth Scope、参数隔离和稳定性—可塑性门
协同工作，但不替代原有晋升证据。

## 4. Soccer 实施

新增模块：

- `training/recovery_foundation.py`：三专家合同、本体门控、29DoF 混合、固定训练分布、
  守门员后继状态、受控跌倒双安全剖面和 S49 快照路由审计。
- `training/recovery_baseline_adapter.py`：HumanUP 与 HoST 的 23DoF 动作适配合同。
  二者不能共用映射：HumanUP 是相对默认姿态的 12腿+3腰+8臂；HoST 是相对当前姿态
  的 12腿+1腰+8臂+2腕。适配器补成 29DoF，省略关节保持当前姿态，超限不裁剪冒充
  正常动作，而是返回 no-op 并拒绝。
- `training/recovery_reference_catalog.py`：仓库提交、许可证、入口、身体、仿真后端、用途
  与限制的机器可审计清单。
- `training/recovery_baseline_evidence.py`：严格聚合四卡物理报告，检查设备唯一性、模型/
  body/scene/交接合同一致性和零硬件权限。

训练 reset 分布被冻结为：20% 真实技能后状态、30% 物理扰动、20% 随机姿态、15%
扑救中间态、10% 最难失败记忆、5% nightmare。该分布已形成可验证的精确整数分配，
但还没有被宣传成已完成大规模 Recovery PPO。

## 5. 外部项目与论文审计

本地项目目录：`/code/rosclaw/rosclaw_football/repos`。

| 项目 | 固定提交 | 许可证 | 当前用途 | 已知边界 |
|---|---|---|---|---|
| HumanUP | `7516e0f` | Apache-2.0 | G1 recovery baseline | 23DoF，supine/prone 分开训练，无随仓权重 |
| HoST | `70bb580` | MIT | G1 多地形/多姿态基线 | prone 可训练；supine+prone 联合训练仍是上游 TODO，无随仓权重 |
| HiFAR | `5a5cef7` | Apache-2.0 | 课程与受控跌倒参考 | 原身体是 Booster T1，不能直接当 G1 策略 |
| AMP_mjlab | `6c7a294` | 未声明 | 统一运动/恢复 AMP 架构研究 | 只读参考，不复制或合并代码，不生成可复用派生实现 |

论文已下载到 `/code/rosclaw/rosclaw_football/papers`，包括 StableMimic、HumanUP、HoST、
State-Dependent AMP、FIRM、PTDL、HiFAR、RoboNaldo、PAiD 和 Humanoid Goalkeeper。

HumanUP/HoST 依赖旧 Isaac Gym，当前机器具备现代 Isaac Lab 与 MuJoCo-Warp，但没有旧
`isaacgym` Python 包；两个仓库也没有提供可直接评测的 checkpoint。因此目前已完成
源代码/许可证/关节动作合同接入，尚未把它们写成“已跑通物理基线”。

## 6. 实验结果

### 6.1 S49 真实快照门控审计

- 样本：50 个真实扑救事件后的因果快照；
- 初版：高动量 31 个中有 6 个过早路由；
- 校准后：
  - `AIRBORNE_OR_HIGH_MOMENTUM -> ABSORB`：31/31；
  - `STANDING -> ATHLETE`：7/7；
  - `PRONE -> GET_UP`：1/1；
  - `KNEELING_OR_SUPPORTED`：8 个仍有明显动量进入 Absorb，3 个进入 Athlete。

这是静态四帧重复的离线路由审计，因为 S49 档案没有连续四帧接触力；证据中已经写明
`STATIC_DUPLICATE_NO_TEMPORAL_CLAIM` 和 `OFFLINE_ROUTING_AUDIT_NOT_PHYSICS_ROLLOUT`。

### 6.2 四卡 R0 物理复测

| GPU | 世界数 | 扰动 | 最终稳定 | 末尾连续稳定范围 |
|---|---:|---:|---:|---:|
| cuda:0 | 4 | ±0.10 | 4/4 | 7.70–8.66 s |
| cuda:1 | 4 | ±0.10 | 4/4 | 7.88–8.48 s |
| cuda:2 | 4 | ±0.10 | 4/4 | 7.70–8.94 s |
| cuda:3 | 4 | ±0.10 | 4/4 | 7.68–8.72 s |
| 合计 | 16 | ±0.10 | **16/16** | **7.68–8.94 s** |

物理后端是 MuJoCo-Warp、无 auto-reset，四个进程分别自描述 `cuda:0/1/2/3`。模型、
源动作、G1 body、scene 与热交接配置哈希完全一致。

### 6.3 当前明确未解决

- S49 真实 failure-terminal/settled post-save 状态上的完整恢复仍是 0/9；
- 当前只有 R0 supine 神经起身专家，缺少 prone、left/right-side 与 kneeling 专家；
- Absorb 目前只有正确的路由/安全/数据合同，没有训练完成的 29DoF 策略；
- 三专家尚未在同一个无重置物理回合中完成连续软混合；
- 尚无“射门—扑救—落地—起身—再次守门”的新视频或晋升证据。

## 7. 验证

- ROSClaw Core `tests/continual`：79 passed；
- Core 新增定向测试：14 passed；
- Soccer S47/S49/S50 轻量回归：20 passed，4 skipped（轻量 venv 无 Torch）；
- Soccer GPU 环境 S47：3 passed，1 skipped（缺 ONNXRuntime，仅跳过数值对照）；
- Soccer 新增 S50 定向测试：13 passed；
- Soccer 全包 Ruff：通过；
- Soccer 全包 mypy（142 个源文件）：通过；
- Core 全包 Ruff/mypy：通过；
- 四卡 A6000 MuJoCo-Warp 真实物理步进：16/16；
- 硬件授权：0。

## 8. 下一执行批次

1. 训练 29DoF `Absorb`：从 S49 高动量快照和失败条件 Dream 开始，只优化冲量耗散、
   头部安全和可恢复后继价值，冻结扑救、Get-Up 与 Athlete。
2. 建立 prone/left-side 专家：以 HumanUP/HoST 的奖励与课程思想重新实现到现有
   MuJoCo-Warp/MJLab 栈，不复制未许可 AMP_mjlab 源码；先在 settled S49 9 个状态上
   达到组件通过，再扩到随机姿态。
3. 将启发式门控蒸馏为四帧本体神经门控，训练时可见专家价值，部署时仍禁止 reference
   phase 和 task truth；门控必须通过时间连续性与专家抖动考试。
4. 三专家闭环后用 `GOALKEEPER_READY` 严格计分，并与 R0、HumanUP-style、HoST-style、
   State-Dependent-AMP-style 做相同种子、相同快照、相同安全剖面对照。
5. 组件通过后才接回动态传球、跑动一脚射门和飞扑，最后录制无 teleport/reset 的完整
   宣传视频；失败回合保留在 Memory/Dream，不删掉做成功剪辑。

## 9. 证据索引

目录：`/code/rosclaw/rosclaw_football/evidence/s50-successor-state-growth-v1`

- `recovery-reference-audit.json`：外部仓库提交/许可证/入口审计；
- `s49-recovery-gate-audit.json`：50 个真实快照的本体门控路由；
- `r0-getup-gpu0-perturb010.json` … `gpu3`：四卡原始物理报告；
- `r0-getup-4gpu-aggregate.json`：16 个世界的严格聚合报告。

当前最重要的科学结论不是“已经 100% 恢复”，而是把两个看似矛盾的结果分开了：
R0 在自己的 supine 邻域内是 16/16，但在真实扑救产生的 prone/side 分布上仍是 0/9。
后续 Growth 必须扩展吸引域与后继状态，而不是继续在已经会的简单邻域刷成功率。
