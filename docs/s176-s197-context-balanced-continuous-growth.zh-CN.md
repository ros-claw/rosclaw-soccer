# S176–S197：上下文平衡连续 Growth 与父代安全闭环实施报告

日期：2026-09-04  
分支：`codex/s142-perceptive-target-routing`  
边界：`SIM_ONLY`、CPU MuJoCo 物理真值、无真实机器人/ROS topic/DDS/串口/CAN/厂商 SDK 指令

## 1. 结论先行

这一阶段打通了一个完整且可审计的自进化闭环：

1. 从 298 条既有完整物理记忆出发训练连续 finisher-plan critic；
2. 发现原 19 维 critic 在整上下文交叉验证中不能泛化，拒绝；
3. 改为上下文等权、整上下文分层四折、选定因果子空间和归一化 RBF；
4. 连续策略第一次上线后由 4/6 退化到 3/6，拒绝；
5. 以失败上下文为中心执行多阶段主动边界搜索，最难场景由完全失败推进到 16 次中 13 次严格成功，最佳落点误差 0.00438 m；
6. 将新成功/失败轨迹写回肌肉记忆，得到 345 条轨迹、66 条严格成功、279 条失败、12 条精确进球记忆；
7. 在已消耗开发集上，离散父策略由旧基线 1/6 提升到 6/6；
8. 连续子策略曾出现“总门通过、但单场精度退化”的盲点，因此新增直接父代物理对照和逐场精度不退化门；
9. 将连续失败反馈写回 critic，并对重复退化的两个上下文建立局部熔断，由已验证父策略接管；
10. 最终 S197 达到 6/6 严格链路、6/6 安全、6/6 精确重放、5 个进球 + 1 次扑救、3 个落点误差不超过 0.10 m，且所有父代不退化门通过。

这证明了“失败能被记录并改变下一代行为”，也证明了 Stability–Plasticity 的工程闭环：探索可以失败，但失败不会覆盖当前最优父策略。

同时必须明确：S197 是**已消耗开发集**，不是 sealed fresh holdout。连续 critic 在新上下文中是否能稳定超过离散父策略，仍需下一阶段冻结后验证。

## 2. 最终可视化

- 视频：`/code/rosclaw/rosclaw_football/videos/s197-context-safe-continuous-growth-final-development.mp4`
- Manifest：`/code/rosclaw/rosclaw_football/videos/s197-context-safe-continuous-growth-final-development.json`
- 规格：H.264，1920×1080，30 fps，1498 帧，49.93 秒
- 视频 SHA-256：`sha256:8914c9cfe80db371bf6e7ebeb35f5046fdbfa785cad3db8640c491bc3320b7fc`
- Manifest 哈希：`sha256:57509e77786a7d2136fa98efc98705f3ec35fd8e652695ae2d30083ab3e11b27`
- 视频只读取通过校验的候选 NPZ 轨迹；像素从不参与评分。

视频覆盖全部 6 个严格通过场景，而不是只挑选最好看的成功片段。画面中的传球者、射手、门将和足球位于同一个 MuJoCo 求解器；没有开球后的 pose/ball teleport。

## 3. 最终量化结果

最终报告：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s197-continuous-finish-plan-v3-final-parent-safe-development/exam-report.json`

| 指标 | S197 子代 | 直接父代 S189 | 早期冻结基线 |
|---|---:|---:|---:|
| 严格链路成功 | 6/6 | 6/6 | 1/6 |
| 安全 | 6/6 | 6/6 | 6/6 |
| 精确进球（误差 ≤ 0.10 m） | 3 | 3 | 未作为父代门 |
| 进球 | 5 | 5 | — |
| 扑救 | 1 | 1 | — |
| 精确重放 | 6/6 | 同场物理对照 | — |
| 最佳进球落点误差 | 0.0043827 m | 相同 | — |
| 进球落点误差中位数 | 0.0954838 m | 相同 | — |
| 最坏逐场父代落点退化 | 0.0 m | 参照 | — |

关键承诺：

- S197 报告哈希：`sha256:a14bcb6488acdf9863f551c645f610ed75a89a433b61c29156f582af4b804c6d`
- S197 实现哈希：`sha256:4027efb172181cb0391c1f52d788ea49e00c617346d2b52799673ee488ff2b0d`
- S189 离散父 actor：`sha256:8792384c44740c6049b98e6a7d3ca9079a7998e7501f58a8d75eb5f75e19690f`
- S195 连续子 actor：`sha256:995023d35837079c7f5e2d65041ee4df1b4446ffa203a435a16766b3ba9c9cbf`
- S195 训练报告：`sha256:592c3273eef795edef5972d01ed54cfc4988f7c48e2d5f34155d3842e7384b6f`

逐场结果：

| Case | 结果 | 落点误差 | S195 路由 |
|---|---|---:|---|
| `s170.precision.v2.00` | 进球 | 0.44144 m | 风险上下文熔断，父代接管 |
| `s170.precision.v2.01` | 进球 | 0.46382 m | 精确离散记忆 |
| `s170.precision.v2.02` | 进球 | 0.08665 m | 精确离散记忆 |
| `s170.precision.v2.03` | 进球 | 0.09548 m | 精确离散记忆 |
| `s170.precision.v2.04` | 门将扑救 | 不适用 | 风险上下文熔断，父代接管 |
| `s170.precision.v2.05` | 进球 | 0.00438 m | 主动边界搜索形成的精确记忆 |

第 00、01 场虽为严格进球，但离期望目标仍分别有约 0.44 m、0.46 m 误差；它们是下一阶段精度训练的主要对象，不能用最佳 4.4 mm 样本掩盖。

## 4. 失败如何转化为成长

### 4.1 Critic 结构失败

S176 v1 使用全 19 维输入的 kernel ridge。整上下文交叉验证结果约为：安全 AUROC 0.374、意图脚 AUROC 0.618、严格成功 AUROC 0.594，未过门。

诊断表明，少数上下文拥有大量局部 probe，会在逐行训练中压倒其他上下文；高维欧氏距离也让 RBF 在稀疏数据上失去可辨识性。

改造后：

- 每个物理上下文总权重相等；
- 四折验证以完整 `context_hash` 为单位隔离；
- fold 分配显式平衡罕见 unsafe、错误接触脚和 strict-success 标签；
- critic 只读取预先声明的因果子空间：传球者 yaw、stance x/y、swing scale 和目标脚 vx/vy/vz；
- 使用归一化 RBF smoother；
- 对二分类头做训练 fold 先验校正；
- AUROC 用未校正平衡分数评估，运行概率用先验校正，避免混淆排序能力和概率校准。

S191 在 345 条轨迹、17 个上下文上通过全部校准门：

- 上下文平衡安全 AUROC：0.6722；
- 意图脚 AUROC：0.6275；
- 严格成功 AUROC：0.7351；
- precision MAE：0.2833；
- post-contact stability MAE：0.0986。

### 4.2 主动边界搜索

最难的 `s170.precision.v2.05` 不是靠一次大网格“碰运气”，而是逐步缩小因果边界：

| 阶段 | 搜索变量 | 严格成功 | 最佳误差 | 结论 |
|---|---|---:|---:|---|
| S180 | arrival timing | 0/16 | — | 实际 phase 不变，否定 timing 假设 |
| S181 | 全 stance envelope | 0/16 | 1.2335 m | stance x 正边界明显改善 |
| S182 | stance 边界组合 | 0/15 | 0.9160 m | x=0.12, y=0.06 更接近球门 |
| S183 | stance edge refine | 0/16 | 0.8838 m | contact frame 256 最优 |
| S184 | contact edge refine | 1/16 | 0.4615 m | 首次严格通过 |
| S185 | target velocity coverage | 5/16 | 0.3085 m | 横向目标速度是关键变量 |
| S186 | target velocity edge | 9/16 | 0.2613 m | 找到局部响应区间 |
| S187 | 0.0025 m/s bracket | 7/16 | 0.07486 m | 首次进入 0.10 m |
| S188 | 0.0005 m/s ultrafine | 13/16 | 0.00438 m | repair 数据门通过 |

S188 报告哈希：`sha256:caa8e261c502b986ad96942ec4d946956115e58a616820e0f9863eeea8d6295c`。

实现新增了可复用搜索策略：`LOCAL_OR_MICRO`、`STANCE_COVERAGE`、`STANCE_EDGE_REFINE`、`CONTACT_EDGE_REFINE`、`TARGET_VELOCITY_COVERAGE`、`TARGET_VELOCITY_EDGE_REFINE`、`TARGET_VELOCITY_BRACKET_REFINE`。它们作用于通用高层 finisher action，不包含某个 case 的硬编码动作。

### 4.3 Credit assignment 修复

旧记忆分数给球速 3 分、精确命中只给布尔 1 分。S188 中 1.367 m/s 动作只比 1.366 m/s 快约 0.003 m/s，却因球速项选中了 0.0747 m 误差动作，而不是 0.00438 m 动作。

新评分保持安全和完整链路的优先级，将球速权重降为 2，并在 0.50 m 目标包络内给予连续精度 credit（最高 2）。已有父代 298 条记忆保持原分数不变，新评分只作用于新增轨迹，避免重写历史语义。

### 4.4 直接父代门与上下文熔断

S192 连续提案从父代 6/6 降到 5/6，但因为早期评测只与 1/6 基线比较，仍被旧门判为通过。新增直接父代四路物理对照后，子代必须同时满足：

- 严格成功数不下降；
- 安全数不下降；
- 精确球数不下降；
- 进球数与扑救数不下降；
- 每个父代进球的落点误差不恶化；
- 候选本身仍为 6/6 确定性重放。

S193 经一次失败反馈恢复到 6/6，但第 00 场误差从父代 0.441 m 恶化到 1.445 m。新增逐场误差门后，该轮被重新识别为退化。S195 将本批实际执行的两个连续上下文写入局部熔断表，在归一化特征距离内回退不可变父策略。最终 S197 所有父代门通过。

这不是把连续学习删除，而是把连续学习的权限改成：**能证明不退化时探索；重复失败的局部由已验证父技能接管；新上下文仍可在 critic 支持域中尝试。**

## 5. 代码改造

- `runtime_finish_plan_actor.py`
  - 上下文平衡 RBF critic 的选定输入子空间、归一化权重和类别先验；
  - 旧 actor 序列化兼容；
  - 连续失败动作排除；
  - 失败上下文局部熔断与父代回退。
- `continuous_finish_plan_growth.py`
  - 上下文等权与分层整上下文四折；
  - AUROC/MAE/Brier 的上下文平衡和逐行诊断；
  - 二分类先验校正；
  - 反馈报告重新按当前父代契约分类；
  - 失败动作和失败上下文写回新 actor。
- `continuous_finish_plan_repair.py`
  - 目标上下文选择；
  - 多阶段 stance/contact/velocity 主动搜索；
  - 修复小预算时 seed 被前几个动作耗尽的问题，改为交错 seed。
- `runtime_finish_plan_growth.py`
  - 连续落点精度 credit；
  - 非有限/负物理指标 fail-closed。
- `runtime_finish_plan_exam.py`
  - 精确球、最佳/中位落点指标；
  - 直接父 actor 同场物理轨迹；
  - 五项父代不退化门和逐场落点门；
  - 父代轨迹文件与报告绑定。
- `continuous_finish_plan_development_video.py`
  - 不再硬编码旧 S175 的 4/6、1 个精确球；
  - 标题、manifest 和 validator 动态绑定当前报告指标；
  - 可渲染全部严格通过案例。

本轮生产代码没有写入本机绝对路径；外部证据和视频仍留在 Git 仓库外。

## 6. 验证结果

环境确认有 4 张 NVIDIA RTX A6000。critic 拟合可使用本地计算资源，但最终物理权威固定为 CPU MuJoCo，以保证确定性和证据可比性。

- `python -m compileall -q src tests`：通过；
- `ruff check .`：通过；
- 本轮改动文件 `ruff format --check`：通过；
- `mypy --config-file pyproject.toml src`：338 个源码文件通过；
- 本轮专项：24 passed，3 skipped；
- 配置当前外部证据根后的联动测试：11 passed；
- ROSClaw Core 1.2.0 临时 home 只读 smoke：`status --json` 返回 `HEALTHY`，7 个核心模块均为 `HEALTHY`，`app list --json` 正常；
- 全量 pytest：912 passed，18 skipped，11 failed。

全量 11 个失败均来自已经安装在本机的 S78/S79/S80/S93/S104–S114 历史外部证据：其实现或依赖 closure 与当前源码不一致，validator 按设计 fail-closed。没有普通功能单测失败。全库 `ruff format --check` 另报告 68 个历史文件会被当前 formatter 重排；本轮未批量改写这些无关文件。

## 7. 当前能力边界

已经证明：

- 完整传球→接球→射门/扑救链路能够从失败物理轨迹中学习；
- 精确动作可固化为内容绑定肌肉记忆；
- critic 的校准单位是完整上下文，而非相关性过强的 probe 行；
- 连续探索失败会改变下一代路由；
- 子代不能凭“比很旧基线强”绕过直接父代退化；
- 所有控制仍保持高层、SIM_ONLY，关节力矩只归冻结神经 actor。

尚未证明：

- S195 在全新 sealed 上优于 S189；
- 第 00、01 场达到 0.10 m 精度；
- 当前连续层在已熔断上下文产生比父代更优的新动作；
- 真实机器人安全与 sim-to-real；
- 端到端在线 actor-critic 直接更新关节力矩策略。

## 8. 下一阶段建议

1. 冻结 S189/S195 及 S197 实现哈希，先登记一组未见的 receiver lane、摩擦、传球 yaw 和球位组合，再开启 sealed holdout；
2. 对第 00、01 场分别做横向速度 bracket 与 stance/contact 二阶段搜索，目标由“严格进球”提高为每场误差 ≤0.10 m；
3. 把上下文熔断从二值表升级为带失败密度和置信恢复条件的 trust region，只有新证据足以推翻失败边界时才重新开放；
4. 增加 actor-vs-parent 的逐场 stability 指标，包括射手尾部摆动、支撑脚滑移、pelvis 最低高度，避免只守住足球结果；
5. 新鲜集若通过，再将连续层进入更大范围的异构域随机化；失败则继续写回，但不能动 sealed 集；
6. 训练直接力矩 actor-critic 时继续保持独立 shadow/teacher、不可热替换和父代回滚，不把本阶段高层 actor 的证据外推为电机级安全证明。

## 9. 通俗解释

可以把当前系统想成“球员 + 教练 + 队医式保险”：

- 离散肌肉记忆保存已经在物理场上成功过的完整动作；
- 连续 critic 像教练，尝试在相邻动作之间调得更顺、更准；
- MuJoCo 比赛是真正的考试，不看视频画面打分；
- 如果教练的新建议让球队比上一代差，建议不会覆盖原动作；
- 同一个局部连续失败两次后，系统会暂时让老动作接管，而不是继续冒险；
- 失败记录仍保留，之后遇到新证据可以重新学习。

因此，本轮最重要的不只是“出现了一个 4.4 毫米进球”，而是 ROSClaw Soccer 已能把失败变成可追溯的新数据，并且在持续学习时不遗忘已经掌握的团队技能。
