# S4a：足球真实滚动修复与证据门控

日期：2026-08-12

## 结论

用户观察到的“足球像方块一样横移”是真实物理缺陷，不是视频错觉。旧三机传球段的
球速中位数约为 1.264 m/s，但球面滚动速度中位数只有约 0.0015 m/s，滑移率约
99.9%。根因是球自由关节把 `0.02` 的线性阻尼错误地同时施加到旋转自由度；对足球
很小的转动惯量而言，这会在约 0.1 秒量级抹掉自旋，而球仍继续平移。

S4a 将编译后球自由关节改为单位明确的两组参数：

```text
平移阻尼：[0.02, 0.02, 0.02] N·s/m
旋转阻尼：[0.00002, 0.00002, 0.00002] N·m·s/rad
```

并新增直接读取六维球速的滚动真实性门控。旧三机证据现在会因
`sliding rather than rolling` 失败关闭，不能再只凭平移速度连续性晋级。

## 同状态 A/B 证据

对照实验从旧三机轨迹中提取传球脚触球后同一帧的足球位姿、线速度和角速度，在同一
CPU MuJoCo 球场中只改变旋转阻尼；两个分支各自再次严格重放。

| 指标 | 旧配置 | 修正配置 |
| --- | ---: | ---: |
| 中位滑移率 | 65.72% | 0.78% |
| P95 滑移率 | 74.03% | 1.62% |
| 中位平移速度 | 0.649 m/s | 1.292 m/s |
| 中位球面速度 | 0.231 m/s | 1.286 m/s |
| 可评估距离 | 0.622 m | 2.584 m |
| 滚动帧占比 | 0% | 100% |
| 判定 | FAIL | PASS |

机器证据：

`/code/rosclaw/rosclaw_football/evidence/s4a-rolling-audit-v1/rolling-audit.json`

同屏 1080p 视频：

`/code/rosclaw/rosclaw_football/evidence/s4a-rolling-video-v1/s4a-football-rolling-before-after-1080p.mp4`

视频侧车：

`/code/rosclaw/rosclaw_football/evidence/s4a-rolling-video-v1/s4a-football-rolling-before-after-1080p.json`

视频为 1920×1080、30 fps、324 帧、10.8 秒、H.264。左右画面使用完全相同的初始
状态、地面摩擦和时间步长；黑色球面标记让旋转肉眼可见。视频像素不参与上述指标。

## 新增工程能力

- `rosclaw soccer physics rolling-audit`：从哈希绑定的外部 evidence 中提取实测传球
  状态，生成旧/新 A/B 严格重放、滑移率曲线和机器判定；
- `rosclaw soccer media rolling-audit`：生成证据下游同屏对照视频；
- `measure_rolling_authenticity()`：领域无关于具体球员动作的足球滚动指标，检查球面
  速度、滑移速度、中位/P95 滑移率、滚动帧占比和地面接触样本；
- 三机 evidence validator 现在强制包含 `ball_velocity` 并复算传球滚动真实性；
- 球门/足球合同升级至 v8，线性和角向阻尼分别哈希绑定。

## 边界

本阶段只修复足球物理基座，没有声称传球者或门将已经训练完成。旧 S3 三机视频应被
视为发现缺陷的历史回归样本，而不是仍然有效的物理通过案例。S4b 必须在修正后的球场
重新生成三角色轨迹、证据和视频。
