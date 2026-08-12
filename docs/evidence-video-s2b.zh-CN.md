# S2b：证据下游视频闭环

日期：2026-08-12
边界：`SIM_ONLY`、CPU MuJoCo 物理权威、无 ROS/DDS/串口/CAN/厂商 SDK、无硬件命令

## 本阶段结论

本阶段把冻结实现中的 G1 任意球视频生成链迁移进 `rosclaw-soccer`，并把“每轮有视频”固化为工程完成标准。视频不是重新摆拍，也不参与评分：它只读取 S2 严格重放已经产出的轨迹；JSON 证据、NPZ 轨迹和渲染器均由 SHA-256 绑定。

阶段视频：

`/code/rosclaw/rosclaw_football/evidence/s2b-free-kick-video-v3/s2b-free-kick-development-1080p.mp4`

侧车清单：

`/code/rosclaw/rosclaw_football/evidence/s2b-free-kick-video-v3/s2b-free-kick-development-1080p.json`

## 视频与物理结果

| 项目 | 实测结果 |
| --- | ---: |
| 分辨率 / 帧率 | 1920×1080 / 30 fps |
| 帧数 / 时长 | 644 帧 / 21.466667 s |
| 编码 | H.264 |
| 助跑距离 / 峰值速度 | 3.472 m / 1.524 m/s |
| 助跑到触球衔接 | 0.946 s，触球前停顿 0.000 s |
| 射门峰值速度 | 11.248 m/s |
| 球门平面目标误差 | 0.02043 m（0.10 m 阈值内） |
| 踢后后退 / 跌倒 /饱和 | 0.000 m / 否 / 否 |
| 最终关节速度 RMS | 0.000747 rad/s |
| 严格重放 | 是 |

必须区分两个结论：球在球门平面命中所声明的 0.10 m 目标圈，但旧 Core 聚合晋级门还要求网内捕获误差和“声明球门角”距离通过；本轨迹的对应值分别为 0.4588 m 和 2.5135 m，因此来源证据总判定仍为 `passed=false`。视频全程显示 `DEVELOPMENT · NOT PROMOTED · SIM ONLY`，本阶段没有声称候选晋级。

## 实施内容

1. 新增有界、`allow_pickle=False` 的轨迹加载器；对时间、骨盆、29 个关节、足球位姿做形状、有限值、严格递增时间和非零四元数校验。
2. 新增平滑位姿采样，四元数采用 SLERP；插值只改变观看采样率，不改变物理轨迹。
3. 新增 `rosclaw soccer media free-kick`，统一生成 720p/1080p 阶段视频。
4. 生成前验证证据域、物理权威、硬件声明、Body 哈希、场景哈希、轨迹文件哈希和轨迹内容摘要。
5. 生成后用 `ffprobe` 复核分辨率、帧率、帧数和时长，侧车清单记录视频、证据、轨迹和渲染器哈希。
6. 调整镜头：球门慢镜头从场内正视目标和实际过线点；触球后切回近距离跟随 G1，保留完整恢复尾段，不能用远景掩盖后退或抖动。
7. 清单额外记录 Python、NumPy、MuJoCo、FFmpeg 与 FFprobe 版本；源码目录、原始证据和输出路径执行 fail-closed 边界校验。
8. MP4 和关键帧保存在源码外，Git 仓库只提交生成器、测试和报告。

## 人工关键帧复核

已检查助跑、触球、球门慢镜头、精确过门线、恢复过程和最终姿态。抽帧保存在：

`/code/rosclaw/rosclaw_football/evidence/s2b-free-kick-video-v3/qa-frames`

最终 v3 修复了首版球门近门柱遮挡目标的问题；恢复段的 G1 也从远景小目标改为近距离跟随，能直观看见躯干、手臂和支撑腿的余振。

## 完整性摘要

| 对象 | SHA-256 |
| --- | --- |
| 视频 | `f9dfccebb22acb55b3e0f67ab78e039a59543641dd5d3e6f0d0a6b5133caf45a` |
| 视频清单文件 | `688bae1876e66748792babbfb0ce1bc887fe546a0a1c24e9f79f298bf48168cb` |
| 来源证据文件 | `132f019359c209516be3b5d7baa15871674c332056815e29c33ee262e9369391` |
| 轨迹文件 | `51b3e890349fdd1b33c427142f375b0c3919ae6a8b659571c1f674a253a6c86a` |
| 轨迹内容摘要 | `74425dbfefb6fdd379c2c993f6e9dc2ee4dc05d23423e011f95033fb9d74773d` |
| 本版渲染器 | `b27efe32b9ce15cd23485296c7a374adae8e087ede3549eace2e57509aa7b61f` |

## 验证

- ROSClaw 精确入口：`python -m rosclaw.entrypoint soccer media free-kick ...`，通过。
- 全量 `pytest`：91 passed、1 skipped（仅旧实现迁移奇偶性测试）。
- `ruff format --check` / `ruff check`：通过。
- 本阶段新增媒体模块严格 mypy（隔离依赖追踪）：通过。
- 全仓 strict mypy 当前有既有 NumPy 裸 `ndarray` 类型债务；未用忽略规则伪装通过，也未把 19 个无关模块的大范围类型迁移混进本阶段。

## 下一阶段

沿同一视频与证据合同迁移多人耦合世界：精准传球者、连续接球射门者和反应式门将共享标准球、标准球门与恢复控制。验收仍先看量化证据，再看同一轨迹视频；视频必须包含传球完整滚动速度、射门者恢复和门将动作尾段。
