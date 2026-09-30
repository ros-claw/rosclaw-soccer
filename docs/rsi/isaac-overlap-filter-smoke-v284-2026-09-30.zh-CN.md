# Isaac PhysX 碰撞组 v284：隔离可建，但单球路不等价

状态：**诊断烟测完成、批量因果训练 REJECTED / SIM_ONLY**。
协议见 `docs/rsi/protocols/overlap-filter-smoke-v284.json`。
按 [Isaac Lab 官方克隆文档](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/cloning.html)
所述，用 PhysX `filter_collisions` 为每个 G1/球环境建立过滤组，
共享地面作为全局组。报告记录场景反向过滤开关和每个环境组的
`includes`/`filteredGroups` 关系；两个 G1/球可以在同一原点启动。

用同源 runner、G1、SONIC、两个静态课程做了四条物理回放：
8 m 分离双环境、0 m 重叠且碰撞组过滤双环境，以及两个各自
位于原点的一机一球对照。原始数据在
`/code/rosclaw/rsi-overlap-filter-v284-{separated-v2,overlap-v2,static-lane0,static-lane1}/`。
正式审计 `/code/rosclaw/rsi-overlap-filter-v284-audit.json`，哈希
`sha256:64791db0bc0c813847bc792649b212b808fef773e681445deb37392ae3f89265`。

| 课程 | 重叠过滤双环境 / 原点单环境：触球后 60 帧前向 | 横向 | 结论 |
| --- | ---: | ---: | --- |
| lane 0 | 2.049 / 2.072 m | −0.406 / −0.422 m | 通过 0.10 m 门 |
| lane 1 | 1.755 / 1.821 m | +0.343 / +0.124 m | **横向差 0.219 m，失败** |

两路触球帧和碰撞体一致，骨盆安全；但仍有单球路差异。当前数据
不能区分是 PhysX 批量求解顺序、SONIC 运行时细微初态差、过滤
后残留跨环境作用，还是临界碰撞的混沌放大；不能声称碰撞过滤
已经解决批量物理等价。**不能将其作为独立反事实监督标签**。

如果未来用于大规模 RL，应明示为带数值扰动的训练域，把策略在
独立进程和新鲜球位的真实物理结果作为最终晋升门，而非要求
每个训练 rollout 与单路逐值相同。这个新训练协议尚未建立，
此实验没有授予训练或宣传视频资格。
