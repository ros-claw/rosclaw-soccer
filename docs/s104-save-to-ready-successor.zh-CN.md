# S104：从“扑到球”到“还能继续守”

## 结论

S104 把 S103 的四角神经扑救接到了一个新的严格后继状态：

> 传球 → 射门 → 真实手套扑救 → 落地吸能 → 主动走回门区 → 连续就绪 1 秒 →
> 接受新的左右横移命令 → 再次连续进入 `GOALKEEPER_READY`。

四个高球角、左右手套、每条两次严格重放全部通过。全程使用同一个 MuJoCo
世界、同一个球和三台同时存在的 G1；没有 reset、teleport、像素评分或硬件命令。

这解决的不是“视频结尾站久一点”，而是技能组合中的终止语义问题：保存动作成功以后，
上一个策略不能把一个仍在晃动、背向球场或无法再次横移的身体交给下一个策略。

## 根因与修复

### 1. 固定出生朝向被误当成实时身体朝向

早期恢复器用 G1 出生时的 yaw 做世界坐标到机体坐标的变换。扑救以后浮动基座已经旋转，
这个固定值会把回位和横移命令投到错误方向。S104 改为从 MuJoCo 浮动基座四元数实时计算
yaw，并对非有限、形状错误和退化四元数 fail closed。

### 2. “原地零命令”只能等晃动消失，不能主动卸力

旧路径在接触以后一直发送零速度。它能最终站住，却不会主动纠正门深、回到门区中心或用
脚步卸掉冲量。S104 新增显式、默认关闭的恢复路径：

- 先保留 2 秒接触后吸收窗口，让扑救和落地包络完整释放；
- 用冻结的 RoboNaldo locomotion foundation 做门深纠正；
- 用有限速度的中心回归把扑救冲量转成可控步态；
- 在门心 ±0.15 m 的小中心袋内停止横移，避免阈值边缘反复切换；
- 用实时 yaw 做朝向闭环；
- 只有同时显式启用稳定器、恢复器和有界 probe，才允许二次横移测试。

这些字段默认均不激活，因此原有 S103 和旧调用方的行为保持不变。

## 严格后继状态门

首次就绪和二次横移后的最终就绪都要求连续 1 秒、50 个控制帧满足：

| 指标 | 门限 |
|---|---:|
| pelvis height | ≥ 0.70 m |
| upright projection | ≥ 0.90 |
| root linear speed | ≤ 0.25 m/s |
| root angular speed | ≤ 0.50 rad/s |
| bilateral support | 100% |
| facing field cosine | ≥ 0.90 |
| inside keeper region | 100% |
| hand tracking error | ≤ 0.18 rad |

中间还必须实际接受一个新的 ±0.14 m/s 横移命令，命令覆盖率不少于 90%，产生同号物理位移、
峰值横向速度不少于 0.03 m/s，横向加速度能力不少于 0.50 m/s²。视频不参与任何一项评分。

## 四角结果

| lane | 手套 | 新命令 | 实际位移 | 峰值横速 | 峰值横加速度 | 最终线速 | 最终角速 |
|---|---|---:|---:|---:|---:|---:|---:|
| left-inner | left | +0.14 m/s | +0.0328 m | 0.2167 m/s | 2.1192 m/s² | 0.00019 m/s | 0.00113 rad/s |
| left-outer | left | +0.14 m/s | +0.0367 m | 0.2041 m/s | 2.0428 m/s² | 0.00017 m/s | 0.00050 rad/s |
| right-inner | right | -0.14 m/s | -0.0164 m | 0.0514 m/s | 0.7450 m/s² | 0.00025 m/s | 0.00117 rad/s |
| right-outer | right | -0.14 m/s | -0.0173 m | 0.0483 m/s | 0.7210 m/s² | 0.00011 m/s | 0.00066 rad/s |

四条最终窗口都是 100% 双脚支撑，原 S103 的真实手套接触、真实腾空、落地和关节安全门
也全部保留。左右响应幅度仍不完全对称，这是后续 S105 要继续优化的诚实缺口，不能用放宽
门限或镜像视频掩盖。

## 可复现证据

- CPU MuJoCo 证据：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s104-save-to-ready-successor-v2/evidence.json`
- report hash：
  `sha256:bbc316bb89f177c4f8d35583ccd9471823bdc348a0d34205652cf0b62ff888d4`
- implementation hash：
  `sha256:a89acfeae4053e51c7f61e7defd514cfff74868fc11ffde994e4e9ca7d04d5ff`
- 41.53 秒、1920×1080、60 fps 可视化：
  `/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/s104-save-to-ready-showcase-v3/s104-save-absorb-reengage-ready.mp4`
- video hash：
  `sha256:de905395a7afb4c8efbb942b43605495d30defbf42a565acccb467319fe02224`
- manifest hash：
  `sha256:143ed8af6ddb4976328c5ed2a50e923c7263d57df980c2dc11c825a1064d4c2c`

视频采用门将本体中心机位展示恢复和二次横移；它是内容绑定的展示材料，不是通过门的证据源。

## 对 ROSClaw 通用能力的贡献

足球只是验证载体。S104 的可复用部分是：

1. **Skill successor contract**：技能的成功包含“下一技能可执行”，而不是只看当前奖励事件。
2. **Measured-frame recovery**：恢复控制必须使用实时本体状态，不能复用出生坐标假设。
3. **Absorb → Re-engage → Ready exam**：对任何带冲击的运动技能，都可以插入新的因果命令，
   验证身体是否真正恢复了可控性。
4. **Fail-closed evidence binding**：配置、Body、模型、许可、父证据、代码实现和轨迹逐层内容绑定。
5. **Safe authority boundary**：学习残差和恢复器没有电机授权；当前结论严格停留在 `SIM_ONLY`。

## 下一步

S105 不应再继续拉长等待时间，而应缩短扑救接触到首次 `GOALKEEPER_READY` 的延迟，并处理
右侧横移响应弱于左侧的问题。推荐把本轮轨迹切成 `impact/capture/center/ready/probe/re-ready`
阶段数据，用冻结 locomotion prior 上的低权限 residual 或 phase-conditioned MoE 学习更对称的
回位步态；晋升仍必须通过本 S104 的完整后继状态门。
