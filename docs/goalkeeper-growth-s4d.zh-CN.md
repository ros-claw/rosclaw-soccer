# S4d：三智能体门将安全封堵学习闭环

日期：2026-08-12

## 本轮突破

本轮没有继续给门将叠加视觉补丁，而是把守门员变成真正拥有独立成功语义的学习角色：
同一个 passer → shooter → goalkeeper CPU MuJoCo 世界中，门将从冻结攻击轨迹的成功/失败
rollout 里搜索有界动作，并以真实接触、扑救、碰后球速和硬安全成本选择候选。

最终候选使用射手本体相位和球状态，在射门前开始横移，随后平滑叠加 0.265 rad 的低位跨步
残差。它在不放宽关节或扭矩边界的条件下首次完成真实扑救：

| 指标 | 结果 |
| --- | ---: |
| 传球距离 / 接球误差 | 2.847 m / 0.0235 m |
| 射门至封堵距离 | 5.176 m |
| 射门峰值球速 | 9.074 m/s |
| 门将触球时间 | 8.192 s |
| 碰前 / 碰后峰值球速 | 8.370 / 5.736 m/s |
| 碰后球速比 | 0.685 |
| 门将横移 | 0.592 m |
| 门将最低骨盆 | 0.729 m |
| 关节越界 / 扭矩越界 / 饱和 | 0 / 0 / 0 |
| 严格重放 | 通过 |

反事实基线使用相同的攻击、同样的早期横移，但关闭低位封堵：球照常进门，且门将没有接触球。
因此扑救来自学到的动作候选，不是降低射门难度或更改球轨迹。

## 修正的物理语义

旧实现只检查球是否穿过门线的 x 平面，球从门柱外穿过也会被记成进球；有状态球网还会在
门框宽高之外捕获足球，视觉上就像一堵透明墙。本轮新增完整球体的门框口径判定：只有球心
保留一个球半径余量、完整进入门框才算进球，球网也只在有效网口内施力。攻击进球和门将扑救
现在是互斥、独立的物理事件。

## ROSClaw 闭环

新命令：

```bash
python -m rosclaw.entrypoint soccer academy train-goalkeeper-block \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output-dir /code/rosclaw/rosclaw_football/evidence/s4d-goalkeeper-block-v3 \
  --source-checkout /code/rosclaw/rosclaw_soccer_s2
```

它依次完成：资产资格检查 → 8 个有界候选 rollout → 安全成本过滤 → 最小安全动作选择 →
候选严格重放 → 无封堵反事实 → 内容哈希绑定 evidence。搜索空间也已把门将的封堵起始帧和
髋关节残差纳入通用三角色 Growth 参数向量。

本轮 evidence 通过开发门，但明确标记 `PASSED_DEVELOPMENT_GATE_NOT_PROMOTED`；它尚未在
多射门位置、多目标和独立 holdout seeds 上完成泛化验收，因此没有被冒充为已晋级策略。

## 证据与视频

- 最终证据：`/code/rosclaw/rosclaw_football/evidence/s4d-goalkeeper-block-v3/`
- evidence SHA-256：`70e4dac36a0740a183691c353f7c050757285c37cdcd60c442685217453821fc`
- 轨迹摘要：`92fe380ff93dc05e10ef473f38138f5261f86d6b59f4eedda0eec27fe16fc44d`
- 1080p 视频：`/code/rosclaw/rosclaw_football/evidence/s4d-goalkeeper-block-video-v2/s4d-three-g1-learned-goalkeeper-save-1080p.mp4`
- 视频 SHA-256：`e725a1dd630f9d5e0bb8b88ec032de03b04c219417a3487a85bbe7be8f26a06a`

视频为 1920×1080、30 fps、38.3 秒、1149 帧 H.264，包括完整连续段和扑救慢动作。
底栏明确显示开发门通过但尚未晋级，像素只用于展示、不参与评分。

## 下一阶段

下一阶段应扩展为 matched attack holdout：随机左右目标、高低球、不同射门时序和速度，训练
门将的侧步、脚挡、手挡和侧扑动作路由；同时保持攻击方策略冻结，用每个角色的 difference
reward 防止多智能体共同退化。只有 holdout 扑救率提高、攻击基线难度不下降、零安全成本和
严格重放同时成立，才允许三角色策略原子晋级。
