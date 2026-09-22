# 滚动接球协议 R1：设计（2026-09-22）

本文是"滚动接球新协议"路线的正式设计。路线决策由用户在复核
`receive-to-next-action.md` 末节证据后作出：冻结 R0 课堂内接球→下一动作
已被证明为耦合封锁（钉住 + 跟随包络 + 球权/深度/摆腿时序互斥），不再继续
在冻结课堂内寻找单一环节修复。

## 1. 证据基础（只列已核实事实）

- R0 课堂协议（`r0_receiving_configuration`）显式开启
  `loose_ball_capture_hold=True, hold_sec=0.6`：接球后身体被钉住 0.6 s。
  比赛世界默认 `loose_ball_capture_hold=False`，没有这一钉住。
- 教师终局偏好"最死的球"已被重定向评分替代并物理验证：同一候选族、同一
  权限下，球可以 0.2-0.3 m/s 保持滚向进攻方向（m0/redirect-receive-dribble-
  v1/v2，控制组精确复现）。
- S208 已在连续比赛世界完成接（滚动球）→ CAPTURE→ORIENT→PLANT→STRIKE
  →真实射门链（PASS_PHASE_CONDITIONED_STRIKE_SAVE）。
- 相位链两个已核实的 SHOOT 取向缺口：`_controller_strike_stance_metrics` 与
  `_strike_tracking_yaw_error` 的方向都来自
  `(goal.plane_x_m, goal.target_y_m)`；`strike_phase.begin_capture` 仅对
  `MatchRole.FINISHER` 开启；射门租约仅授予 SHOOT 意图。
- PASS 激活路径只有 owned_option（触球后 0.2 s）与 prospective
  （`prospective_enabled=True` 且球权为 None 且当前追球者）；PASS 无租约。

## 2. 协议目标与边界

目标：在显式新协议 R1 中证明 接球（球保持滚动）→ 相位接续 → 真实传出脚
触球 → 队友到达 的完整链，先在极少量已见场景，再逐步扩课。

边界（与 R0 相同纪律）：

- SIM_ONLY；CPU MuJoCo 为最终物理真值；视频不作成功标签。
- 考试不放宽：接球考试、身体安全、`pass_contact_chain`、带球后继考试原样
  复用；相位状态机单调前进、超时/失稳 fail closed。
- R1 是新协议，不是放宽 R0：R0 课堂与 40/128 路线门原样保留为历史参照；
  R1 课程银行独立生成、独立资格验证，不与 R0 成绩拼接比较。
- 教师可用特权仿真信息做可达性，不是部署策略；学生训练（DAgger/PPO）必须
  等 R1 教师资格之后，且一次仅一个球员 plastic。

## 3. 协议组成

### 3.1 世界配置

以连续比赛配置为底（`default_continuous_match_config`，无接后钉住），
显式声明的 R1 差异：

| 项 | R1 选择 | 依据 |
| --- | --- | --- |
| 相位控制器 | 开启（`StrikePhaseConfig`，S208 同款，含 0.36 s 触球预测） | S208 已验证 |
| 相位捕获角色 | 扩展到接球者（不再仅 FINISHER），按场景显式绑定 | 接球者可能是组织者 |
| 相位朝向/站姿目标 | 传球任务目标感知（见 3.2） | 球门取向对 PASS 不适用 |
| 踢球桥 | `pass_enabled=True, bilateral_enabled=True, task_context_bound=True`，默认入场几何不变 | 与既往 PASS 探针一致 |
| PASS 激活 | 首选 prospective（球权 None + 当前追球者）；不新增 PASS 租约 | 不新增所有权语义 |
| 球权保持 | `contact_possession_hold_sec=0.20` 不变 | 不延长球权 |

### 3.2 传球目标感知的相位接续（唯一计划内的源码改动）

为 `_strike_tracking_yaw_error` / `_controller_strike_stance_metrics` 增加
显式任务目标注入：当持球决策为 PASS 且存在
`decision.target_position_m` 时，相位度量与朝向误差以该目标替代球门；
SHOOT 路径与默认值逐位保持不变（冻结 S208 路径）。改动限 soccer 仓、带
定向测试（默认路径哈希不变、PASS 目标注入路径的新几何正确性、禁止硬件
权限），走正常 review+commit。

### 3.3 接球教师

沿用接触教师（S208 的 `default_phase_strike_teacher` 保持了已验证的传球
接法）。R1 的接球终局目标是"卸力后球仍以 0.15-0.35 m/s 滚向下一动作
方向"（重定向评分已物理验证可达），而不是钉死。教师权限边界不变：
有界 12D 残差、原滤波、原速率、原安全投影；私有预测预算显式声明。

### 3.4 考试与证据

每次实际执行必须同时给出：原接球考试结果、相位状态机轨迹（单调性）、
身体安全、`pass_contact_chain` 的完整 500 Hz 触球归属、控制见证、
独立重复全数组一致、介入前前缀与归档对照一致。允许失败；不允许把
"请求了 PASS"或"相位到达 STRIKE"当作传球成功。

## 4. 里程碑与判据

| 里程碑 | 要证明的事 | 退出判据 |
| --- | --- | --- |
| MR1 | 单一已见比赛场景：滚动接球→相位→真实传出脚触球→队友到达 | 2 次独立重复均通过 pass_contact_chain 且原接球考试保留；否则记录并停止该机制 |
| MR2 | 方向/速度小课程簇（≤8 个新场景）同链 | 每场景重复一致；无安全回归 |
| MR3 | R1 课程银行（新 seed 生成，含 DEV/SEALED 分离）资格 | preflight 全部通过；R0 历史参照不动 |
| MR4 | 稳定反馈教师资格（闭环重规划，非播放） | 教师在新场景重复达标后才谈 DAgger |
| MR5 | DAgger 学生→asymmetric PPO（一次一个球员 plastic） | 按总纲配对考试与 3 seed 规则 |

任何一级失败：记录、停止该机制、回到上一级判据；不允许跳过 MR1 直接
扩课或开训。

## 5. MR1 有界探针预定义

- 场景：连续比赛夹具中的一个已见传球→接球场景（S208 同族），接球者
  改为请求 PASS（目标为当时测得的队友）。
- 机制：3.1+3.2+3.3 的全部；prospective 激活；无租约。
- 预算：≤6 次实际执行（2 对照复现 + ≤4 候选）；每次私有预测 ≤9000；
  全部冻结源码并哈希绑定；先写 protocol.json（假设/验收/停止判据）。
- 验收：对照精确复现；候选两次独立重复均通过 pass_contact_chain、
  原接球考试保留、身体安全、相位单调。
- 停止：任一候选不安全/接球考试丢失 → 否决；两次均无传出脚触球 →
  否决并审计相位几何；不做无界参数扫描。

## 6. 已知风险与开放问题

- 相位接续在 PLANT 收敛的是"预测触球点站姿"；传球目标移动时预测误差
  可能使 STRIKE 无法到达——MR1 正好检验这一点，不预设成功。
- prospective 依赖"当前追球者"仲裁；队友/对手更近时接球者可能拿不到
  prospective 资格。若发生，显式记录，不作为失败掩盖。
- 低滚动摩擦（≈0.02 m/s²）下球不会自停；相位链按设计处理滚动球，
  但踢球激活后的 0.6-1.2 s 摆腿延迟内球仍在动，传球精度是真实风险。
- R1 与 R0 成绩不可拼接；R1 课程银行未建成前，任何"成功率"都仅限
  已见小样本，不作为泛化结论宣传。

## 7. 当前状态

协议设计完成（本文）；3.2 的源码改动与 MR1 探针尚未实施。没有正在
运行的实验；没有训练、Sealed 或晋升。
