# S78：前锋—门将交替最佳响应闭环

日期：2026-08-24
状态：**首个射门—扑救交替成长周期完成；开发证据通过，未使用 sealed holdout，禁止晋升**

## 结论先行

本阶段没有把“前锋进球”和“门将扑救”做成两个互不相关的视频，而是在同一个 CPU
MuJoCo 三 G1 世界、同一颗物理足球和同一目标下完成了一次交替最佳响应：

```text
父代前锋射门
  → 父代门将真实触球并扑出
  → 冻结门将
  → 新前锋改变触球相位/瞄准，精准进球
  → 冻结新前锋
  → 新门将根据失败轨迹前移并下探
  → 真实触球、球未越过门线、扑救成功
```

正式 v4 矩阵包含 7 个发现对局和 2 个严格复演对局，并在 S79 物理实现加固后重新生成。最终结果：

- 父代对局：门将扑出，前锋失败；
- 前锋 best response：进球，落点误差 `0.002 m`，射速 `9.876 m/s`；
- 门将 best response：真实触球并阻止过门线；
- 前锋最低骨盆高度 `0.667 m`；
- 最终门将最低骨盆高度 `0.724 m`；
- 两个决赛轨迹严格复演一致；
- 非有限状态、关节/力矩越界、执行器饱和、硬件命令均为 0。

这是一个**开发可达性周期**，不是泛化能力晋升。目标、物理世界和候选曾用于开发，尚未
使用封存射门方向、高度、摩擦和初始状态；报告强制
`promotion_eligible=false`、`sealed_holdout_used=false`。

## 1. 先修复真实集成断点

最初启动统一射门—扑救剧集时，物理尚未开始就 fail-closed：共享世界控制器仍要求旧的
球形 `reach-envelope` 名称和尺寸，而统一球场已经升级为真实成人手套尺寸
`0.21×0.11×0.07 m`。两份合同不一致，导致门将场景无法解析。

本轮没有绕过校验，而是让共享世界绑定球场实际生成的左右手套 geom，并验证：

- 名称为 `goalkeeper_left/right_goalkeeper_glove`；
- 手套必须连接到对应 wrist body；
- 半尺寸必须精确为 `(0.105, 0.055, 0.035) m`；
- 任何名称、身体或尺寸漂移继续 fail-closed。

修复后，第一个完整 15 秒基线真实发生传球、射门和进球：射速 `9.074 m/s`、目标误差
`5.51 cm`，前锋和门将均站立。它给后续对抗学习提供了可解释的父代状态。

## 2. 前锋如何从被扑出中成长

父代 `goalkeeper-g0-upright-block` 对原低角度射门能够稳定扑救。先尝试 5 个脚部偏航
扰动，结果 5/5 被扑出；这说明继续在同一触球拓扑上调小参数没有突破价值。

随后改变策略参考高度，球没有真正变成高球，却改变了触球相位和横向出球路径。该负面
事实被保留为“并未学会高球”，但它发现了一个新的低位远角动作族。重新声明实际物理
目标并严格复测后：

| 策略 | 对父代门将 | 目标误差 | 射速 | 前锋最低骨盆 |
|---|---|---:|---:|---:|
| `striker-g1-yaw-only-failure` | 被扑出 | 无落点 | `9.942 m/s` | `0.673 m` |
| `striker-g1-contact-phase-escape` | **进球** | `≈0 m` | `9.876 m/s` | `0.667 m` |

新前锋不是靠提高球速硬冲；核心变化是接触相位和策略瞄准，使同一冻结门将没有形成球接触。

## 3. 门将如何从失球中成长

冻结新前锋后，原先 8 个 hip-pitch 幅度全部失球，证明失败不是腿再抬高一点就能解决。
轨迹诊断发现球到门前时约为 `(y=1.35,z=0.19) m`，右手约为
`(y=1.15,z=0.71) m`：门将横向接近，但低位下探不足。

分解实验得到清晰的 successor-state 路线：

| 门将候选 | 物理结果 | 最低骨盆 | 解释 |
|---|---|---:|---|
| `depth-only-failure` | 触球后仍进球 | `0.732 m` | 前移让球进入可达域，但偏转不够 |
| `downreach-only-failure` | 触球后仍进球 | `0.734 m` | 下探增加偏转，但方向合同不完整 |
| `low-dive-a` | **扑出，未越门线** | `0.724 m` | 前移 + 肩俯仰 + 肩外展形成有效低扑 |
| `low-dive-b` | **扑出，未越门线** | `0.736 m` | 较保守外展也成功 |

激进肩动作曾使骨盆降到约 `0.07 m`，即使有接触也被安全门拒绝。最终选中的不是动作最大
的候选，而是完整扑救成功、保持站立并通过所有关节/力矩门的 `low-dive-a`。

## 4. ROSClaw Growth 闭环

新增 `training/shot_save_league.py`，把一次性调参升级成可复用的对抗 Growth 合同：

1. 分别哈希前锋与门将策略，禁止把两名智能体合成一个不可追责参数包；
2. 冻结对手后评估角色自己的 best response；
3. 前锋门要求真实传球、真实脚触球、进球、误差 ≤0.10 m 和身体安全；
4. 门将门要求真实门将—球接触、球未进、最低骨盆 ≥0.65 m 和身体安全；
5. 被扑出的射门及“触球仍失球”都写入 Failure Memory；
6. 攻方和防方决赛分别重新运行，物理结果与完整轨迹 digest 必须逐字一致；
7. 只有两侧 best response 都成功才标记 cycle complete；该标记不等于 promotion。
8. 验证时重新计算当前实现哈希；代码与证据实现不一致时，即使证据自身哈希完整也拒绝。

这套合同不是 G1 足球专用的 Core 能力。抽象后就是：

```text
冻结对手/环境冠军
→ learner 产生安全 best response
→ 冻结 learner
→ 对侧 learner 从失败 successor state 反制
→ 双方失败进入不可变 Memory
→ disjoint holdout 决定是否晋升
```

它可以复用于抓取—脱手对抗、导航—扰动对抗和操作器—物体动力学共同成长；足球策略、
球场和角色实现仍留在 downstream Soccer 仓库。

## 5. 证据与视频

正式证据目录：

```text
/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/shot-save-league-v4/
```

关键文件：

- `shot-save-growth-round.json`
  - report hash：`sha256:0559339dac55dedd5218ebadaeb8741fa89f0ca1658b1f772621c84746f48a28`；
  - implementation hash：`sha256:74281d0ee0640a48bae1a29008b45f4b7f7da6d60d044f33dfc0babac12faea8`；
  - 文件 SHA-256：`936ac19e3c418c1e33a9e1c995d479def2a4183531b5311bc174b82c65e10192`；
- `attacker-best-response-trajectory.npz`
  - SHA-256：`bafed838fd353dbe9123996eb15303a31644c9ad9af92b886f3491a775ace3ee`；
- `defender-best-response-trajectory.npz`
  - SHA-256：`51464b3894e52fb6d300da244d3d50775870221a2bcd81eca0e06a6d1ac4d34d`。

视频：

```text
/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/shot-save-league-v4/
  shot-save-growth-v4.mp4
  shot-save-growth-v4.json
```

- 18 秒，1920×1080，30 fps，H.264，540 帧；
- 先显示新前锋突破冻结门将，再显示新门将扑出完全相同的射门；
- 每个对局都包含慢动作物理复演；
- 视频 SHA-256：`8fce6a84b3d431e8173737ffc07eaabff539f917ac0eaedbbea8558c36538ead`；
- manifest 自哈希：`sha256:efb8fa051561c626d10fe7c5c60e5f61376128b717aa7c6d21e40e297b827584`；
- `visualization_only=true`、`pixels_used_for_scoring=false`。

## 6. 不能越过的结论边界

当前可以说：

> 在同一个 CPU MuJoCo 三 G1 世界中，前锋和门将完成了一次有真实球—脚/球—身体接触、
> 角色独立策略、失败回流和严格复演的交替最佳响应周期。

当前不能说：

- 前锋已经学会高球；本轮是低位远角；
- 门将已经达到 80% 未见射门扑救率；
- 策略已通过方向、高度、摩擦、延迟和初始状态 holdout；
- 本轮使用了端到端大规模 RL；当前是可解释小策略族的物理 Growth；
- 扑救后倒地恢复已经接回本轮低位站立扑救；本轮门将没有倒地；
- 可以部署到真实 G1。

## 7. 下一轮硬门

1. 将当前 CPU league 扩展为四卡 MJWarp/MJX 批量 dojo：攻方目标族覆盖左/右、低/中/高，
   防方覆盖站立挡球、跨步、低扑和高扑；CPU MuJoCo继续作为独立物理真值。
2. 为每个角色训练 recurrent actor-critic，而不只搜索少量参数；actor 只能读取部署可见
   的球历史、本体感和接触，critic 可以读取特权状态。
3. 使用 population league 和 prioritized failure replay，避免双方只对单一冠军过拟合；
   同时冻结旧冠军回放，解决 stability-plasticity dilemma。
4. 预先封存方向、高度、球速、摩擦、延迟和初始站位 holdout，候选生成阶段不可读取。
5. 对会倒地的高扑，将 S77 OpenTrack→Capture→locomotion oracle 接到同一连续世界：
   `shot→save→land→recover→goalkeeper-ready→second shot`，禁止 reset/teleport。
6. 最终晋升要求双方都在新对手 population 上提高，而不是仅在一条零和对局里轮流获胜。

## 8. 最终质量门

- `pytest`：射门—扑救证据、真实 G1 手套碰撞体、三人媒体、角色学习和联合策略搜索共
  `43 passed`；
- `ruff`：本轮涉及的源码与测试全部通过；
- `mypy --strict`：新增 league、媒体及改动后的共享世界模块全部通过；
- `compileall`：三份运行源码全部通过；
- 证据 validator、视频 manifest validator、SHA-256 与 `ffprobe` 均重新核验；
- 视频为 H.264、1920×1080、30 fps、540 帧、18 秒；
- 全程 `SIM_ONLY`，未连接 ROS/DDS/vendor SDK，未发送硬件命令。
