# S6：MotionDecode + OmniContact 组合模仿闭环

日期：2026-08-12
边界：`SIM_ONLY`、CPU MuJoCo、无 ROS/DDS/厂商 SDK/硬件命令

## 结论

本轮把两类互补数据接到同一个三智能体、单足球的物理闭环：MotionDecode v6
继续负责全身姿态与速度协调；OmniContact 只用官方 `train` 切分中 24 个高质量右脚
球接触事件，负责触球附近的右腿运动形状。Growth 搜索 4 个小权限候选，并以 S5
通过候选作为不可遗忘的父技能。

最终候选通过开发门但不晋升：`PASSED_DEVELOPMENT_GATE_NOT_PROMOTED`。它保留
0.02 的 MotionDecode 姿态/速度权重，只给 OmniContact 0.0025 的位置残差权限。
数据教师没有直接输出力矩，仍通过 PD、关节预测保护、85% 力矩权限投影和硬力矩上限。

| 指标 | S5 父技能 | S6 组合候选 | 变化 |
|---|---:|---:|---:|
| 6.64 m 射门误差 | 0.01765 m | 0.01900 m | +1.35 mm |
| 踢后支撑脚滑移 | 0.05685 m | 0.04805 m | -15.5% |
| 躯干最大侧倾 | 0.23061 rad | 0.22421 rad | -2.8% |
| 触球窗关节加速度 RMS | 91.55 rad/s² | 90.24 rad/s² | -1.4% |
| 踢后关节加速度 RMS | 49.90 rad/s² | 48.10 rad/s² | -3.6% |
| 尾段晃动指数 | 0.0000499 | 0.0000472 | -5.4% |
| 踢后峰值后向速度 | 0.00174 m/s | 0.00571 m/s | 仍低于 0.01 m/s 门限 |

射门速度 9.04 m/s，传球误差 2.33 cm；无跌倒、无关节/力矩/执行器违规；
足球滚动审计和双重严格复演通过。

## 失败如何帮助成长

第一版直接把 OmniContact 的绝对关节角混入已有专家。触球加速度虽略降，但滑移从
5.68 cm 恶化到 9.31 cm，侧倾从 0.231 rad 增到 0.266 rad，后向速度增至
0.0184 m/s。根因是两个控制器的姿态原点不同：绝对角教师把已经稳定的髋部和支撑
几何拉走了。

修复不是删掉数据，而是把 OmniContact 第一帧对齐到当前专家，只迁移接触过程中的
相对位移；再把权限压到 0.25%。候选中 0.10%、0.15% 和错位接触相位都因滑移、
侧倾、根部加速度或尾段晃动被拒绝。成功和失败都进入机器可执行的稳定—可塑门。

## 数据边界与架构

- `football_motion_prior.py` 新增坐标对齐的右腿位移迁移，不复制数据集绝对姿态；
- `shared_world.py` 支持第二教师、逐关节权限、运行哈希与轨迹中的实际残差；
- `composite_imitation.py` 以 S5 为父技能做多目标反事实搜索；
- `composite_imitation_evidence.py` 绑定 Body、两个教师、训练分区、held-out 承诺、
  候选、环境和轨迹摘要；
- `three_player.py` 在渲染前重算实现哈希、请求/轨迹哈希、滚动与物理门；
- CLI：`rosclaw soccer academy train-composite-imitation`。

OmniContact v1 教师只读取 `train` 文件；`val/test` 只记录不可逆分区承诺，不访问
其指标。原始数据、教师 JSON、轨迹和视频均在仓库外，代码仓只保留算法与报告。

## 证据与视频

- 接触教师：`/code/rosclaw/rosclaw_football/evidence/omnicontact-right-foot-contact-prior-v1.json`
- 证据：`/code/rosclaw/rosclaw_football/evidence/s6-composite-imitation-growth-v2/g1-composite-imitation-growth.json`
- 轨迹：`/code/rosclaw/rosclaw_football/evidence/s6-composite-imitation-growth-v2/trajectory.npz`
- 1080p 视频：`/code/rosclaw/rosclaw_football/evidence/s6-composite-imitation-video-v1/g1-composite-imitation-growth-1080p.mp4`
- 视频：36.73 秒、1920×1080、30 fps、1102 帧、H.264
- 证据 SHA-256：`b28e6f2adc0093b0e01b683962d2aea9af54c76145558dbfe78292edfa6a54f2`
- 轨迹摘要：`ffc531e6d211392965afcb55f3cea2016144b27dcf45c62c96736724272c6256`
- 视频 SHA-256：`467fd3323e7119156fe0972fb0b1fb105651cd81c96834d22db8b068ed015e0f`

## 下一步

本轮证明真实接触数据可以在不破坏精准父技能的情况下改善平衡和冲击，但当前
OmniContact v1 只有 0.36 秒位置教师，改善幅度仍有限。下一阶段应从同一训练事件
提取同步的关节速度与球/脚接触状态，做代表事件而非坐标中位数蒸馏；随后把跑—停—踢
相位扩到约 2 秒，并在同样的精度、滑移、后退和安全门下训练小型神经残差 actor。
