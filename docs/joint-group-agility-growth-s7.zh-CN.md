# S7：关节组灵活性成长与局部鲁棒门

日期：2026-08-12
边界：`SIM_ONLY`、CPU MuJoCo、无 ROS/DDS/厂商 SDK/硬件命令

## 结论

本轮没有把“动作幅度变大”当作灵活，而是把灵活定义为：在不忘记精度、支撑、平衡和
踢后恢复的条件下，腰与手臂能获得独立的数据教师速度权限。运行时将 MotionDecode
全身教师拆成 29 关节位置预算和 29 关节速度预算；腿保持 S6 的 1.00，Growth 在腰、
手臂附近搜索。搜索器最终选中腰 1.14、手臂 1.15，并要求相邻候选形成安全局部盆地，
不能依赖一次偶然的接触轨迹。

最终候选通过开发门但不晋升：`PASSED_DEVELOPMENT_GATE_NOT_PROMOTED`。

| 指标 | S6 父技能 | S7 候选 | 变化 |
|---|---:|---:|---:|
| 6.64 m 射门误差 | 19.00 mm | 8.59 mm | -54.8% |
| 踢后峰值后向速度 | 5.71 mm/s | 2.05 mm/s | -64.1% |
| 踢后支撑脚滑移 | 48.05 mm | 46.75 mm | -2.7% |
| 触球窗关节加速度 RMS | 90.24 rad/s² | 88.87 rad/s² | -1.5% |
| 踢后关节加速度 RMS | 48.10 rad/s² | 47.19 rad/s² | -1.9% |
| 踢后根部加速度 RMS | 2.800 m/s² | 2.717 m/s² | -3.0% |
| 躯干最大侧倾 | 0.2242 rad | 0.2280 rad | +1.7%，低于 0.24 门限 |

射门速度 9.04 m/s，传球误差 2.33 cm；无跌倒、无关节/力矩/执行器违规。局部
邻域 6 个候选中 4 个通过；1.13 因滑移刚超过 6 cm 被拒绝，1.16 因滑移与加速度
同时退化被拒绝。由此确认选中点处于一个窄但可测的安全盆地。

需要如实说明：本轮上肢运动能量只提高 0.066%，可见动作幅度没有显著突破。真正
突破的是解耦架构、稳定性、精度与反事实鲁棒门。宣传视频可以展示更稳定精准的结果，
不能声称已经获得大幅人类式灵活性。

## 失败消融与原因

- 直接提高挥腿速度会改变原专家的离散重复相位，射门误差恶化到约 2 m，甚至丢失
  触球；已拒绝。
- 在策略前拼接恒定前向速度会破坏专家隐藏相位和来球时序，射门失败；已回滚。
- 把 OmniContact 的绝对关节速度直接混入当前策略，会使滑移、侧倾和根部冲击变差；
  原因是数据集坐标/相位与当前专家不一致，当前不激活 v7 速度教师。
- 同时放大腰、手臂的位置教师虽能增加姿态幅度，但会把误差推到 14–15 cm，并增加
  支撑滑移；未进入保留候选。
- 踢后平滑交接 2–8 帧把支撑滑移从约 5.3 cm 放大到 7–20 cm；已拒绝。

这些失败没有被删除：代码把位置/速度逐关节权限分开，并新增“父技能 + 小扰动局部
盆地”门，后续 Growth 不会因一次漂亮视频选择脆弱候选。

## MOSAIC 下一阶段教师

本轮也完成了 MOSAIC 数据资格与蒸馏入口：读取 CDLA-Permissive-2.0 声明，以参考
项目 `csv_to_npz.py` 的明确 29 关节顺序作为内容哈希合同，从 SE52 敏捷梯、SE56
影子拳击、SE57 棒球挥击、SE63 足球快速点球各选一个高能窗口，蒸馏为 SIM_ONLY、
未晋升、内容绑定的速度教师。

后续闭环已完成两步。第一步把四类动作的中位数速度教师接入仅腰臂的 PD 目标速度；
消融表明它会相互抵消语义，达到可见幅度前就增加滑移，未晋升。第二步按任务语义
预先选择 `SE63 soccer_taps`（选择时不读取测试奖励），把速度轨迹积分为首尾归零、
峰值不超过 0.125 rad、仅双臂的姿态残差，并移到踢后随动窗口。正式候选在 9/9
局部邻域通过，手臂行程提高 24.0%，该窗运动能量提高 25.1%，射门误差仍为 8.59 mm，
滑移 4.69 cm，无跌倒。教师始终只经过原 PD/力矩安全链，未直接输出力矩。

## 代码与验证

- `agility_growth.py`：通用关节组候选、灵活度指标、稳定—可塑门和局部盆地门；
- `agility_evidence.py`：搜索、双重严格复演、请求/轨迹/实现哈希和滚动审计；
- `shared_world.py`：位置与速度独立的 29 关节教师权限；
- `mosaic_agility_prior.py`：MOSAIC 许可、形状、时钟、关节顺序及源文件哈希资格；
- CLI：`rosclaw soccer academy train-agility`；
- 媒体与 evidence validator 已支持 S7 schema。

验证结果：`166 passed, 1 skipped`；ruff、format 全通过；本轮 5 个核心模块严格 mypy
通过。整仓历史 numpy 泛型问题不在本轮范围。

## 证据与视频

- MOSAIC 教师：`/code/rosclaw/rosclaw_football/evidence/mosaic-g1-agility-prior-v1.json`
- MOSAIC 足球语义教师：`/code/rosclaw/rosclaw_football/evidence/mosaic-g1-soccer-taps-agility-prior-v2.json`
- 严格证据：`/code/rosclaw/rosclaw_football/evidence/s7-joint-group-agility-growth-v2/g1-agility-growth.json`
- 轨迹：`/code/rosclaw/rosclaw_football/evidence/s7-joint-group-agility-growth-v2/trajectory.npz`
- 1080p 视频：`/code/rosclaw/rosclaw_football/evidence/s7-joint-group-agility-video-v3/g1-joint-group-agility-growth-1080p.mp4`
- 视频：36.73 秒、1920×1080、30 fps、1102 帧、H.264
- 证据 SHA-256：`75963fe75c7bb53cabcaaef4abd8c6e8f46b1cf944b1ce9760ca0c23fe8ef0f2`
- 轨迹摘要：`551742b708d151749b9d02b868b8c451131c065006e6ffc9628982fd5d2b9ebb`
- 视频 SHA-256：`babbe56cefe9a30d14b3aef957f5bb8af84aead7c62794270fc6aa6a03a30189`
- MOSAIC 教师 SHA-256：`619c810f90c412fcd7a2464249c7ca772fcf56f34033aef4cdd3a7f87b8c7691`

- 正式随动证据：`/code/rosclaw/rosclaw_football/evidence/s7-visible-follow-through-growth-v3/g1-follow-through-growth.json`
- 正式随动轨迹：`/code/rosclaw/rosclaw_football/evidence/s7-visible-follow-through-growth-v3/trajectory.npz`
- 正式随动视频：`/code/rosclaw/rosclaw_football/evidence/s7-visible-follow-through-video-v3/g1-visible-follow-through-growth-1080p.mp4`
- 随动证据 SHA-256：`683f5d67217fbcabe12977db6f02a6f75d30bc2ade6f8a90cfe2bd276fbd3150`
- 随动轨迹摘要：`138b34fc0d966876b1d0cea160b5b028a212c2f103816adb0cb09de63e229927`
- 随动视频 SHA-256：`c542450c3b0d68429b694ffcb2f9fe5bf5d04edf165390af451cfcffe0be8c08`

## 下一步

1. 将 MOSAIC 四类动作做当前父策略相位对齐，训练低权限、仅腰臂的 residual actor；
2. 把教师目标从“关节速度更大”改成角动量抵消与双臂—腰—支撑腿协调；
3. 在球位置、来球速度、摩擦和命中目标扰动上跑多情景局部盆地，而不只扰动权重；
4. 只有在可见幅度、精度、滑移、后退和加速度共同改善后，才产出“灵活性突破”视频。
