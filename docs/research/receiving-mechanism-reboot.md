# Receiving Mechanism Reboot：M0 启动记录

2026-09-20。依据用户的 `rosclaw_soccer讨论0920.md` 与 `rosclaw_soccer下一阶段实施总纲.md`，后者的收敛范围优先。不启动 S2781 或另一轮全队 PPO。

## 已落实的方向变化

- 停止 whole-team leg12 PPO / epochs sweep 配置族；不是停止所有研究。
- 当前 17/128 策略命名为 Receiving Champion R0，但明确限定为 **研究参考**，不改变比赛默认模型或晋升资格。
- 后续个人学习一次只允许一名 Plastic 球员，首个研究焦点为 blue.playmaker；本次没有学习更新，八人全部冻结。
- M0 可达性 → E2 闭环接触教师 → E3 学生 → 有界 PPO → fresh sealed 的顺序不能被基础依赖 smoke test 跳过。
- 战术数据准备与物理接口资格验证分开推进。暂不扩展 4v4。

## 本次实际完成

### 1. 冻结研究参考与证据

外部目录 `receiving-mechanism-reboot/r0` 保存模型副本、文件和策略哈希、身体哈希、观测与动作契约、奖励／评分／窗口定义源码副本、原始控制配置和场景哈希。
重新认证 128 堂历史课程原始轨迹及回放，提取 17 堂成功作为 Historical Retention Bank。
它们仍是公开开发资料，不是重新得到的 128 堂新采样，也不是 blind。

另生成 256 条 Development Bank 课程定义，覆盖八角色、四球速、八侧向位置，尚未执行。
这些是课程规格，不是 64 个已恢复的完整物理初态；运动状态、支撑脚、控制器历史还需要专门的快照构造与验证。
Sealed Bank 规定为 256 例，本次没有生成或打开；必须在候选及评分器锁定后进入一次性评估流程。不能把一个普通可读文件名叫 sealed 就声称具备隔离保证。

### 2. 四动作空间对照的校验代码

`training.receiving_reachability` 要求 A0 12D、A1 29D、A2 SONIC、A3 SONIC 加残差四路，在同一 64 个不同物理初态上，采用配对的物理、评分、优化器种子及搜索预算。
缺路、复用证据、重复物理初态、未严格回放、预算超限、NaN 类型伪装等不能作为有效比较。
模块返回有界搜索计数，不授予训练或晋升权限；输入原始证据认证仍由调用方负责。
有界 Oracle 失败只表示该搜索预算下未找到解，不证明动作空间的物理上限。

### 3. 真实基础仿真：SONIC 与 PAiD

本地 SONIC low-latency 和 v1.1 各执行 100 控制帧、1,000 物理步，基础烟测判据通过；再独立重跑一次，两个报告内容哈希相同。
这里证明本地冻结模型、既有适配器和 CPU MuJoCo 能实际协同执行，不证明接球、不证明 A2/A3 超过 A0，也不是完整逐帧轨迹严格回放证书。

PAiD 官方滚动球 ONNX/MuJoCo 示例实际运行四次，种子 92001、球速采样 0.1–0.3 m/s，按其原生目标线进球判据为 3/4。
同种子独立再跑四次，`episodes.csv` 字节完全一致。这不是八个独立场景，也不是 ROSClaw 接球考试。
官方输出 `E_contact_acc` 为 NaN，保留并标注缺测；`is_succ_phc` 为 0，不能把 3/4 进球率说成所有动作跟踪指标都通过。

SONIC 本地参考 checkout 比刚 fetch 的上游 main 落后 14 个提交；本次没有覆盖本地参考文件，也没有宣称现有适配器已采用最新 per-motor scaling。该差异及输入输出契约需在 A2 接口定版前审计。

### 4. 战术数据与能力桥接起步

从官方 SkillCorner Open Data 获取一场完整 tracking，按 Git LFS 指针核对 89,543,442 字节和 SHA-256。
读取 71,451 帧，提取 31,287 对有效半秒帧对，共 407,634 条球员归一化 waypoint 标签。相邻标签相关，不能称为 40 万个独立比赛情境。
`training.skillcorner_waypoints` 使用真实球场长宽；拒绝跨半场、帧间隔错误和重复球员，缺失或外推而非检测到的端点不补零；没有臆造意图类别。
目前只有单场、没有按独立比赛留出、没有进攻方向统一，也没有训练 Human Tactical Prior 或 MAPPO。

`training.receiving_capability` 已从 R0 的真实接球证据生成按角色／球速／侧向分组的计数。未知 readiness 和 duration 保留为空。
单个公开网格的观察频率不是已校准概率；不得直接将 17/128 填成所有 TacticalWorld 接球动作的成功概率。

## 没有完成的部分及接续条件

| 内容 | 当前状态 | 下一步的证据要求 |
|---|---|---|
| 64 个代表性物理初态 | 未完成 | 包括站立／运动、左右支撑及边界情形；保存恢复后的控制器历史和接触状态 |
| A0/A1/A2/A3 Oracle | 未执行 | 四路使用同一组初态和预算，完整实际请求→执行动作→物理响应记录 |
| SONIC 新部署差异 | 已发现，未适配 | 核对 per-motor scaling、关节顺序、参考历史、模型和配置配套 |
| Research Coach 通用状态机 | 未实现 | 复用 Core 的候选、证据及 Plasticity 边界，在隔离工作树开发通用契约 |
| 一次性 Sealed 执行隔离 | 未实现 | 候选锁定、评分器锁定、消费记录持久化、重启后拒绝复用 |
| Contact Teacher / DAgger / PPO | 未启动 | 先通过 E1/E2，不以 smoke test 代替资格 |
| TacticalWorld 2v1 | 尚未运行 | 接入有覆盖和不确定性说明的技能模型；先补齐队伍、进攻方向和意图数据契约 |

本次 Campaign JSON 是有预算的研究声明，不是已经完成的自动研究控制器；没有让元数据自动授权新训练。
Core 当前工作树有既有修改，本次未触碰，不将足球专用逻辑伪装为 Core 通用能力。

## 外部参考与许可边界

本阶段仅使用总纲批准范围。SONIC/PAiD/OmniContact/OpenTrack/SkillCorner/MARLadona/MAPPO+FSP 为清单；没有新增其他算法路线。

- [SONIC 官方模型说明](https://github.com/NVlabs/GR00T-WholeBodyControl)：核实 low-latency 是四帧 SMPL 参考前视，不等于端到端延迟测量；冻结模型，不从头训练。
- [PAiD 官方仓库](https://github.com/TeleHuman/HumanoidSoccer)：本次复用外部官方 sim2sim，未拷贝进 MIT 源码。README 明确 CC BY-NC 4.0，并明确禁止用于商业产品宣传示例。
- [OmniContact 数据卡](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset/blob/main/README.md)：研究数据保持外置，不打包进默认商业发布物；本次未训练其数据。
- [SkillCorner Open Data](https://github.com/SkillCorner/opendata)：MIT，保留署名。实际数据固定到提交 `4340d274572876239c154c90bc507a9b3250a656`。

## 验证

新增两组测试及相关接球契约测试共 65 项通过。全仓 Ruff check、compileall 通过；三个新模块 mypy（`--follow-imports=silent`）通过。
不静默跟踪依赖时，既有模块报告三个类型错误：runtime_finish_plan_actor、mjwarp_contract、goalkeeper_whole_body_reach；未把依赖失败记成全仓通过。本次没有重跑全量 pytest。

原始数据、权重、参考项目、实验脚本和结果在外部目录；Git 只提交自有校验、转换代码、测试和本记录。
当前仍然是 **M0 准备与基础资格验证完成一部分，M0 可达性结论尚未得到**。
