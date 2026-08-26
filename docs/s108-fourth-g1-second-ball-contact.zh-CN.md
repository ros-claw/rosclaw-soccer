# S108：第四台 G1 与第二颗物理足球的真实触球接口

## 结论

S108 没有把 S107 的仿真发射器简单换一个名字，而是先验证了一个更基础、也更容易被假阳性掩盖的问题：能否让第四台 G1 在四机器人、两颗从仿真开始就存在的物理足球场景中，依靠冻结的 RoboNaldo ONNX 策略真实走近并用解剖学脚部碰撞第二颗球。

正式结果为：

```text
四台 G1 + 两颗物理足球从 t=0 同时存在
  → 第四台 G1 使用冻结 ONNX locomotion policy 接近
  → anatomical right foot 首次有序接触第二颗球
  → 球速 0.001 → 7.098 m/s，朝球门方向飞行
  → 前锋保持有限状态、无力矩或关节越界
  → 完整 rollout 严格重放得到完全一致的结果和轨迹摘要
```

正式晋升为 `PROMOTED_SIM_ONLY_SECOND_STRIKER_CONTACT`。

必须明确：S108 只晋升“第二前锋物理触球接口”，`complete_second_save_claimed=false`。它没有声称已经完成“第一扑—恢复—第二前锋射门—第二扑”的整条链，也没有声称第二球命中高死角或被门将扑出。把触球接口先单独考试，是为了避免把球发射器、位置重置或预设初速度伪装成 G1 射门。

## 1. 为什么 S107 的同一颗球不能直接覆盖所有二次来球

最初尝试是保留 S107 的三台 G1 和同一颗活球，把第二次威胁扩展到右内侧球路。这个候选暴露了两个问题：

1. 第一次扑救后，足球可能已经越过球门或落到不适合再次进攻的位置；
2. 若从该位置继续施力，虽然没有写 `qpos/qvel`，本质仍可能变成“从球门后方把球拉回来”的非足球课程。

S108 因此新增 fail-closed 门禁：二次威胁只能从球场一侧的合法发射口袋开始，目标速度必须朝向球门。失败候选被结构化记录在：

`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s108-right-inner-second-save-failure-v1`

它被正确拒绝为 `second-threat live ball is not in a field-side launch pocket`，而不是为了获得通过结果放松空间约束。

同时修复了证据序列化缺陷：没有发生二次接触时，接触高度改为 JSON `null`，不再写非标准 `Infinity`；候选或严格重放抛出 `RuntimeError` 时也会生成结构化 `REJECTED_DEVELOPMENT` case。这样失败仍可进入 growth 账本，而不会留下没有结论的请求目录。

从这个失败得到的架构结论是：连续实战中的第二个进攻者需要第二颗从 `t=0` 就存在的物理球。它让第一次扑救后的球保持原样，同时允许另一个智能体准备下一次威胁，不需要在中途替换或搬运足球。

## 2. 四 G1、两物理球场景

新增 `build_g1_four_player_two_ball_stadium_model`，在统一标准球场和球门内编译：

- 原射手 G1；
- 传球 G1；
- 门将 G1；
- 第二前锋 G1；
- 原有标准足球；
- 第二颗标准质量、半径和摩擦参数的物理足球。

四台 G1 共 `116` 个 actuator，第二颗球有独立 six-DOF free joint，并复用原球与草坪之间的显式 `condim=6` 接触求解参数。测试锁定了机器人数量、球的质量/半径、自由关节阻尼、接触 pair 和非法出生位置拒绝。

### 2.1 修复“足球像方块一样横移”的底层物理根因

开发第二球时出现了一个重要现象：静止圆球会在草坪上自行横移，看起来像方块滑动。根因不是渲染，而是通过 MuJoCo `MjSpec` 动态创建球体并显式设置质量/惯量时，惯性坐标 `body.ipos` 被错误推断成世界出生坐标。地面支持力因此相对质心产生米级力臂和巨大假扭矩。

修复包括：

1. 显式固定第二球的 `body.ipos=(0, 0, 0)`，让质心位于球体自身原点；
2. 使用球体正确惯量 `0.4mr²`；
3. 为第二球增加与原球一致的显式 `second_ball_floor` solver contract；
4. 分别绑定平移和转动阻尼，避免 free-joint 单标量错误地抑制滚动。

回归测试直接检查编译后质心偏置和 contact pair，避免以后场景生成器再次制造“看起来像球、动力学却不是球”的资产。

## 3. 第四台 G1 如何真正把球踢出去

第二前锋没有得到未来球轨迹，也没有直接写入球速度。它使用冻结的 RoboNaldo ONNX locomotion policy 和现有站立/步态接口接近第二颗球；控制输出仍经过动作裁剪、力矩预算、关节范围和安全的触球后 handoff。

开发中没有凭肉眼手调一个看起来接近的位置。系统从 S107 已通过的真实射手轨迹读取“脚部发生接触时，球相对射手根部的局部几何”，得到触球口袋约为：

`(forward=1.285 m, lateral=-0.018 m, z=0.115 m)`

第四台 G1 与第二颗球从仿真开始就按这个独立初始条件存在。考试随后要求：

- 首次有效球体接触必须来自 anatomical foot geom；
- 接触前不得有其他 G1、手套或未知几何体碰球；
- 峰值接触力必须在 `20–1500 N`；
- 接触后球速增益至少 `4 m/s`；
- 球的正向速度至少 `3 m/s`；
- 前锋全程状态有限、骨盆不低于 `0.60 m`；
- 不得出现关节或力矩越界；
- 不得使用 reset、teleport、球速写入或外加扭矩。

策略目标与物理声明目标被刻意分开：前者只是冻结策略的逆标定输入，后者才是球门中的实际任务目标。S108 不以最终落点得分，防止把“能够真实触球”与“已经精准射门”混成同一个结论。

## 4. 正式 CPU MuJoCo 结果

| 指标 | 结果 |
|---|---:|
| 机器人 / 足球 | `4 G1 / 2 physical balls` |
| 第二球从 t=0 存在 | `true` |
| 接触时间 | `7.682 s` |
| 首个合格接触部位 | anatomical right foot |
| 峰值接触力 | `502.226 N` |
| 接触前峰值球速 | `0.000518 m/s` |
| 接触后峰值球速 | `7.097768 m/s` |
| 接触后正向峰值球速 | `6.414995 m/s` |
| 第二前锋最低骨盆高度 | `0.651252 m` |
| 未预期的触球几何体 | 无 |
| 关节 / 力矩违规 | `false / false` |
| reset 或 teleport | `false` |
| 严格重放 | `true` |

严格重放会重新执行完整物理 rollout，并要求结果对象和压缩轨迹摘要完全一致；不是重复读取第一次保存的文件。

## 5. 证据与视频

### 5.1 正式物理证据

- 目录：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s108-fourth-g1-second-ball-contact-v2`
- source commit：`1fbf6ace57249392b02d39c8d2729164a5fc314a`
- report hash：`sha256:31143e69a8057711c95584a1bc04d8bd5597f16344393add1539d2a28163b702`
- implementation hash：`sha256:c5079b73fcbadb8326d1e2207138d3b9d24e08bcc712f279e7ae5eaa56dc8886`
- request hash：`sha256:d64e71ec255ac12608390d06de297f339321bcba6364935385af853e6fe34d73`
- trajectory hash：`sha256:b998a41ee9e9e5faa2d03ea594aa79661be20dee2cfb5ded0165d61ef7e1e5ab`
- trajectory digest：`sha256:d2c63d4f7d75fbba6fa27f93c77dd1a17d4b97bd70008a9c2f669abf66b839e1`

### 5.2 1080p60 多机位证据下游视频

- 视频：`/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s108-fourth-g1-second-ball-showcase-v1/s108-fourth-g1-second-ball-contact.mp4`
- 时长：`19.267 s`
- 分辨率：`1920×1080 @ 60 fps`
- 帧数：`1,156`
- video hash：`sha256:578f2e58b98728121b041b6376465c9c5ae93b31ce5cbaa906691ac676c2f3b6`
- manifest hash：`sha256:40821dc3e9a1383fa7c0ccba63b3223df40431bd87c68a48d336391266d6c741`

视频逐帧重放正式 trajectory 中四台 G1 的根部和 29 个关节状态，以及两颗球的物理姿态。视频清单绑定 evidence、request、trajectory 和视频字节；像素不参与晋升，`visualization_only=true`、`promotion_eligible=false`。

## 6. 对 ROSClaw 通用 growth 的意义

这次产物不是“第四个足球脚本”，而是一个可迁移的多智能体接力接口：

```text
provider A 完成事件
  → 世界状态不重置
  → provider B 在独立物体上产生真实有序接触
  → 接触前缀、来源几何体、能量变化和安全包络被证据化
  → 严格重放
  → 只晋升已经证明的接口能力
  → 下一阶段组合成长
```

它可以迁移到双机械臂交接、移动机器人协作搬运、多无人机接力观察等任务。关键不是把所有能力一次性写进一个巨大 reward，而是先让每个 provider 的输入、输出、物理因果和 authority ceiling 可独立拒绝，再由上层连续任务把这些已证明接口组合起来。

S108 也给 growth engine 增加了两种重要失败记忆：非法空间起点和运行时物理异常都必须成为可读取的拒绝证据，而不能因为没有生成成功视频就从成长历史中消失。

## 7. 诚实边界与 S109

S108 仍保持 `SIM_ONLY`、non-commercial，不授权 ROS/DDS、真机或在线直接关节力矩更新。当前没有证明：

1. 第四台 G1 的第二球能在第一次扑救后的正确时间到达球门；
2. 第二球达到指定高度和横向落点；
3. 门将会把因果观察从第一球切换到第二球；
4. 第二球发生真实手套接触并被扑出；
5. 二扑后再次恢复为 ready。

S109 将在同一四 G1、两球世界中组合 S107 与 S108：第一条传射扑链保持不变；门将真实恢复达到 ready 后，第二前锋的冻结策略继续产生脚球碰撞；observer 在有序触球后建立新的 threat epoch，并只跟踪第二颗球；随后要求第二次 anatomical glove save 和最终恢复。整个链仍必须禁止 reset、teleport、球发射器和直接球速写入，并通过前缀一致性与严格重放。

## 8. 回归验证

- S108 世界、考试、正式外部证据和视频 manifest：`5 passed`；
- S108 涉及源码和测试的 ruff：通过；
- S108 涉及源码和测试的 mypy：通过；
- 排除已被当前实现哈希主动判旧的外部 evidence tests 后，项目回归：`664 passed, 15 skipped, 7 deselected`。

直接运行全部测试时，7 个旧证据校验会 fail-closed：S78、S79、S80、S93、S104、S106 和 S107。它们的外部文件绑定的是生成时的实现，而 S108 修改了共享世界构建或连续考试源码，因此当前 `_implementation_hash()` 与旧报告不再相等。这不是本轮断言或仿真失败，也不应通过更新 JSON hash 来伪装解决；旧证据仍是对应历史 commit 的有效记录，但不再是当前源码的可晋升证据。S108 新证据已经绑定当前实现并单独通过。

结论：S108 已把“第二名前锋能否真实制造下一次威胁”从想法变成了可重放的物理接口，但它故意没有越权宣布完整二扑成功。下一个阶段的突破标准不是更酷的视频，而是把这个已合格接口无缝接入第一次扑救留下的真实连续状态。
