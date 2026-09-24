# RSI-M0 一人一球的物理基线（2026-09-24，SIM_ONLY）

本实验从冻结 SONIC 小脑起步，检验 G1 在 **500 Hz MuJoCo** 中能否真实用脚碰触成人标准尺寸、质量的球，并使整球进入 2.4 × 1.6 m 训练球门。这是 DISCOVERY 基线，**不是**学习完成、能力晋升或可宣传的高水平射门。

## 可复核的执行

- G1 为 29 自由度实体，SONIC 以 50 Hz 提出关节目标；力矩 PD 和球/脚/地面碰撞由 MuJoCo 500 Hz 积分。球半径 0.11 m、质量 0.43 kg，初始位置 (1.5, 0.1, 0.11) m；球门线 x=5.0 m。
- 正向速度 1.2 m/s 的两次不同执行记录在 `/data/rosclaw_overflow/rsi-sonic-ball-contact-20260924-v3-primary` 和 `...-v3-replay`；500 Hz 重建的首次接触为左脚，整球入门，最低骨盆 0.701 m，球速峰值 2.943 m/s。独立验证哈希 `sha256:e5fe6296a6f8974bcdabc64cca9c0cb654b39133f69e019a620af709df1ed14d`。
- 提速至 1.4 m/s 的两次不同执行记录在 `/data/rosclaw_overflow/rsi-sonic-ball-contact-20260924-v3-speed140-primary` 和 `...-v3-speed140-replay`；首次接触仍为左脚，整球在第 151 个 50 Hz 帧进入球门，球速峰值 3.014 m/s，最低骨盆 0.697 m。独立验证哈希 `sha256:f3e0a96296c0b1bcd2e906b16d5a5f2b0343f2155686da4359c0f712b157f5cd`。
- 两组各自的 300 帧控制记录和 3000 帧物理位置逐数组一致；检查器在同一编译模型上从物理位置重建首次机器人—球接触，并独立复算球门线、球速、身体最低高度。测试还检验篡改轨迹、伪造进球帧、同包冒充重放。
- 1.5 m/s 试验暴露了失败：直线会偏出球门；某个横向速度修正虽入门，但首次接触是右小腿，故不能称“脚先射门”，不准用于正向展示。仅这几次场景不代表新鲜场景成功率。

## 视频与边界

720p 开发预览 `/data/rosclaw_overflow/rsi-sonic-ball-goal-speed140-20260924-preview.mp4` 从上述物理轨迹渲染，manifest 同目录 `.json`。镜头与慢动作仅用于观察，**不参与物理评分**。复看后动作仍偏谨慎步行，球是低平球，故目前不把它认定为高质量宣传成片。

该基线没有新的神经权重、没有 actor 训练、没有 Train/Fresh 分离、没有高角度射门和完整传接射。下一步应冻结本基线，在严格分离课程上训练受限触球策略，先比较脚先接触率、球门区误差、峰值球速及失稳，再审计新鲜场景；只有候选相对父策略显著拓展能力且守住安全边界，才能考虑晋升。Isaac 迁移仍未通过，不能把 MuJoCo 成绩外推至 PhysX 或真机。

复核命令：

```bash
PYTHONPATH=src:/code/rosclaw/rosclaw_test/src /code/rosclaw/rosclaw_soccer_s2/.venv/bin/python \
  -m rosclaw_soccer.rsi.verify_sonic_ball_goal \
  --primary /data/rosclaw_overflow/rsi-sonic-ball-contact-20260924-v3-speed140-primary \
  --replay /data/rosclaw_overflow/rsi-sonic-ball-contact-20260924-v3-speed140-replay \
  --stadium-assets /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy
```
