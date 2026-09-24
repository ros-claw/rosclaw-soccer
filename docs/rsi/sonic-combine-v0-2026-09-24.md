# RSI-M0 SONIC Athlete 物理基线（2026-09-24，SIM_ONLY）

本记录是 M0 的 P3 导航/平衡父策略实测，不是 learned Athlete、触球策略或
RSI-M0 完成声明。五项考试和消融均在 Soccer 现有 G1 MuJoCo 模型中实际积分；
原始数据只存外部目录，仓库只保存可复算的实现和校验器。

## 合同与执行

- `SonicAthleteAdapter` 将冻结 SONIC 的 29 维关节目标变成受限 `MotorAction`。
  输入只接受本体 `qpos=36/qvel=35`，显式补齐 SONIC 旧接口所需但网络并不
  使用的球自由度；不向策略泄露球的真值。模型、qualification、增益、
  physics、joint map、20 ms 控制时钟和身体哈希不匹配时锁闭。
- 目前只支持平地的相对速度、转向和停止；射门/触球、绝对航向、改高度或
  非本步意图会拒绝。此处没有 actor 权重更新，也没有机器人执行权。
- `scripts/rsi_sonic_combine.py` 预声明 5 门课程，每门 150 帧、50 Hz SONIC
  推理，500 Hz MuJoCo 积分；同一个初态成对执行完整物理重放，另有一条
  同初态零力矩消融，共 **11 次物理执行**。零力矩对照不是候选策略。
- `verify_sonic_combine.py` 独立校验协议/完成/逐课程 receipt 与轨迹 SHA-256，
  从原始 qpos/qvel 复算高度、姿态、位移、转向、末速和合格判定，验证两次
  重放逐数组相同、消融与 run 同初态，并拒绝篡改轨迹和伪造晋升标记。

## 实测结果

证据目录：`/data/rosclaw_overflow/rsi-sonic-combine-20260924-v2`。
`complete.json` 的 manifest 哈希为
`sha256:c46f0ab229b666a793a6eee412dd20395960829ded49fda72c1d87075b1b5238`；
独立复算的 verification 哈希为
`sha256:9ab62ac50d0bd8d64d4a40281a3f91ab51e942e7a482dfe47153727ef3c4422a`。

| 课程 | 主要实测 | 最低骨盆高度 | 本课程阈值结果 |
|---|---:|---:|---|
| 原地站立 | 位移 0.066 m | 0.748 m | PASS |
| 步行 0.4 m/s | 位移 0.868 m | 0.745 m | PASS |
| 跑动 1.2 m/s | 位移 2.859 m | 0.701 m | PASS |
| 转向 0.8 rad/s | 转角 0.713 rad | 0.746 m | PASS |
| 0.6 m/s 后停止 | 最后 10 帧平均速度 0.033 m/s | 0.724 m | PASS |

五门在该单初态下 5/5；每门第二次完整物理重放的数组逐项完全相同。
零力矩对照骨盆最低 0.079 m、位移 0.354 m；与跑动课程同初态，说明
在这套模型和控制器下，SONIC 输出对稳定跑动有明确物理因果贡献。

验证命令（在 Soccer 仓库根目录）：

```bash
PYTHONPATH=src /code/rosclaw/rosclaw_soccer_s2/.venv/bin/python \
  -m rosclaw_soccer.rsi.verify_sonic_combine \
  --evidence-dir /data/rosclaw_overflow/rsi-sonic-combine-20260924-v2
PYTHONPATH=src:/code/rosclaw/rosclaw_test/src \
  ROSCLAW_SOCCER_RSI_COMBINE_EVIDENCE=/data/rosclaw_overflow/rsi-sonic-combine-20260924-v2 \
  /code/rosclaw/rosclaw_soccer_s2/.venv/bin/python -m pytest \
  tests/test_rsi_sonic_combine_verify.py -q
```

## Isaac 对照与下一门

Isaac Lab 官方 G1 USD 是 **43 关节**（手部 14 关节），而 SONIC/MuJoCo
接口是 29 关节。官方资产 4 环境、120 个 120 Hz 步的默认位置控制后骨盆
0.376–0.404 m，左髋目标跨环境差 0.500 rad，终态差 0.394 rad；此前本地
URDF 转 USD 的 29 关节 G1 也下沉。因此这两项只证明
资产/关节/球可在 PhysX 加载和响应，不证明 SONIC 迁移失败，也不证明资产
损坏：两者都尚未进行 29→43 关节映射、动作增益/频率/初态/接触校准。

下一步先固定跨引擎身体映射与站走动作等价条件，再做独立 PhysX 小规模
轨迹审计；接着才是一人一球的 learned contact policy 和真正的 Train→Fresh
对比。当前五门只有一个初态、没有球、没有新权重、没有 Fresh/Sealed，不能
据此晋升或称为自进化。完成状态：**RSI-M0 NOT COMPLETE**。
