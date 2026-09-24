# Physical RSI M0 架构审计（2026-09-24）

## 基线与目标

- Soccer `main` 在审计开始时为 `c67df1b69204337dc79c38ad24cc4f21fc50e615`，工作树干净，远端同提交。
- R1 唯一确认的完整传接射进球是单一已见物理场景；±2 cm 网格 1/7，广域 0/8。详见 `docs/research/rolling-receive-protocol-r1.md` §17–20 及外置证据 `m0/aim-teacher-calibrate-v1`。四个 seed 并未扰动轨迹，不能算四个独立场景。
- 本阶段科学目标：冻结父代，在预先承诺的数据分区上训练具有实际运动权限的接触策略，证明新鲜场景的成功区域扩大，同时保留身体安全和父代已知成功。
- 截至审计时，R1 没有开启新 DAgger/PPO、Sealed 或新 Champion。本审计不改变这些状态。

## 优先复用的现有代码

| 层 | 已有入口 | 边界与缺口 |
| --- | --- | --- |
| 仿真与控制 | `skills/team/independent_team_world.py`，`skills/team/motor_option.py`，`growth/strike_phase_controller.py` | 500 Hz MuJoCo/50 Hz 控制、身份和执行权限已经存在。R1 成功依赖现有相位与目标配置；在旧世界再建一套动作执行器会失去可比性。 |
| 物理证据 | `skills/team/physics_evidence.py`，`training/pass_contact_chain.py`，`training/dribble_successor_exam.py` | 接触、发送脚到接收脚、后继动作判据可复用。观察接口不能写电机；`world passed` 不等于足球成功。 |
| 运动底座 | `providers/g1/{receiving_sonic,sonic_navigation,sonic_audit}.py`，`skills/athlete_foundation/{backend_catalog,foundation_shootout,motion_atlas}.py` | SONIC/其他底座就绪分级和评测合同已存在。完整统一的实时 AthletePolicy 接口与站/走/转/停/追球的同条件 Combine 尚缺；既有 SONIC 冷交接接球仅保球 1.30 s。 |
| 接触学习 | `docs/s128-s131-neural-contact-growth.zh-CN.md`，`docs/s142-s167-prepared-finish-growth.zh-CN.md` 所对应 neural contact、准备式计划与 Growth 模块 | 这些是不同历史课/资产/协议，不可直接把 3/6 Sealed 或其他局部晋升移植成 R1 的资格。可借鉴动作权限和冻结底座模式。 |
| 个体/球队 | `growth/independent_agent_cell.py`，`skills/team/player_lineage.py`，`growth/alternating_team_growth.py`，`docs/multi-agent-growth-s4b.zh-CN.md` | 独立身份、单 Plastic 边界、配对归因已有基础。M0 先单人一球；未校准的能力模型不能驱动真实团队晋升。 |
| 研究/晋升 | Core Growth/Practice/Memory 和 Soccer `growth/paired_champion_gate.py` 等 | 通用晋升留在 Core。外置完整 episode 和隔离分区需要明确合同、持久化与导入适配。Core PR #587 仍独立审查，不在 Soccer 复制通用 Research Coach。 |
| 参考资产 | `/code/rosclaw/rosclaw_football/repos`、`datasets` | ProtoMotions、SONIC、OmniContact、OpenTrack 等已经存在。外置源 SHA/源码与权重许可必须分别核实；源码树不得自动当作训练 checkpoint 或已验证底座。 |

## 已被实测否决的重复路线

- 冻结 R0 中只延长球权、换接收目标、局部绕球或释放残差，均未形成真实传球后继。
- R1 的成功点对 2 cm 扰动、导航层教师介入不稳。新增手写瞄准和导航扫描只能继续在同一脆弱动作链附近搜索。
- A0 12D 小残差在 R1 再触球尝试中未产生步态相位正确的第二次接触；这不是数学不可能证明，但已不足以支撑继续无界调参。
- 将官方 SONIC 29DoF XML 直接替换到足球场并非合格迁移；armature、增益、接触几何、参考 lookahead 均需单独绑定。见 `reference_audits/sonic-deployment.md`。

## M0 的具体缺口与次序

1. 锁定父代：保存 Soccer/Core SHA、R1 场景/动作/资产/评测配置及历史回执哈希。旧 `aim-teacher` 完成域是 CONSUMED 开发材料。
2. 参考 bootstrap：固定源 SHA 与许可证文本哈希，权重单独记录；不把外部源码或数据复制进 Soccer 包。
3. 统一最小合同：运动目标、身体反馈、动作目标、策略 artifact、物理 episode 和数据分区。合同不得自己授予硬件执行权。
4. 运动底座 Combine v0：在相同 G1 资产和 MuJoCo 物理下测站、走、转、停、追球；使用原底座与 SONIC 的相同考试，不以观感或私有预测排名。
5. Ball Interaction Gym：先一个 G1、一球、指定脚/出球速度/方向，显式新策略拥有的关节目标或受限 latent 权限；小规模 GPU 物理确认后再扩大。
6. 在训练前固定 DISCOVERY/TRAIN/CONSUMED_DEV/FRESH_HOLDOUT/SEALED，取样算法与初态承诺；R1 已见单点及全部手工修复邻域归 CONSUMED。
7. 先有 CPU MuJoCo 父/子匹配物理评价与动作权限证据，再允许晋升；奖励、预测、视频与 Isaac 训练分数不能代替。

## 资源与工程事实

- 当前本机 4×A6000；审计时每卡约 13–14 GiB 被其他任务占用，不结束这些进程。
- `/data` 可用约 40 GB，`/code` 约 122 GB。大型轨迹只写外部证据根，先测容量。
- 已有 Isaac Sim 6.0.1 和一个 Isaac Lab 源码子模块；Soccer 开发 venv **未安装** `isaaclab`/`isaacsim`/`skrl`。源码存在不等于能运行 GPU vectorized G1 任务。须在隔离环境装配并先做一个小批次物理 smoke。
- Soccer `AGENTS.md` 允许审查测试后 direct push `main`。Core 脏工作树和历史证据不得覆盖；Core PR #587 不自行合并。

## 停止条件

父代不能严格复现、资产或 solver 与协议不一致、新 actor 没有实际控制权限、环境物理无效、奖励升而真实球接触不变、训练/新鲜集泄漏、身体不安全、源码/权重 hash 在实验中变化：保存失败原始回执和 `STOPPED.md`，不晋升、不打开下一门。
