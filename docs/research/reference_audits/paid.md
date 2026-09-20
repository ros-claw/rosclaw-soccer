# PAiD / HumanoidSoccer 参考审计（首次实跑）

2026-09-20。参考源码提交 `e72e470230047dedaf66df0983f1d0ab746faeb5`。
来源：[TeleHuman/HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer)。代码与模型保持外置，未复制进本 MIT 项目。

## 本次确认的接口

- reference motion：官方 `motions/soccer-standard` 十条 NPZ；实际读取关节位置／速度和身体位姿／速度。
- action：ONNX 输出 29 维动作，按默认角度与逐关节 scale 映射为 PD 目标；不是直接输出裸关节力矩。
- observation：实测 ONNX `obs=[1,160]`，另有 `h_in/c_in=[2,1,128]`、`time_step=[1,1]`。
- obs 构成：参考关节位置／速度、重力方向、参考角速度、身体角速度、关节相对位置／速度、上一动作、身体坐标系中的球和目标位置。
- recurrent state：每次 trial reset，避免跨场状态污染。官方 `_fit_obs_dim` 有截断／补零兼容路径，未来接入 ROSClaw 时必须改为显式契约核验，不能静默容错。
- controller frequency：本次 `control_dt=0.02 s`，即 50 Hz。
- ball randomization：官方初态采样＋非重叠拒绝采样；本次启用 0.1–0.3 m/s 滚动球，未人为修改运行中的球。
- transition structure：官方 progressive helper 先 motion tracking、后 rolling-ball；本次只执行发布策略推理，不宣称重跑 Isaac Lab 训练链。
- reward：本次没有训练；下方单独审计固定源码中的训练配置，不从 MuJoCo 评价指标反推 reward。

## 可复现命令

在外部源码目录，使用安装了 MuJoCo 与 ONNX Runtime 的仿真 Python：

```bash
python exp/mujoco_soccer_experiment.py \
  --policy ckp/policy_30000.onnx \
  --motion-path motions/soccer-standard \
  --num-trials 4 --seed 92001 \
  --enable-ball-vel --ball-speed-range 0.1 0.3 \
  --output-dir <external-new-output>
```

## 结果与不能混淆的边界

四次原生目标线进球结果为 1、1、1、0；首次脚球接触率 4/4。
四次最大球速分别约 5.228、4.698、4.989、2.964 m/s。这是小规模原生参考 smoke，不是接球成功率、不是标准足球比赛验收、不是跨随机种子的稳定成功率估计。
官方摘要 `is_succ_phc=0`、`is_succ_phc_rel=1`，`E_contact_acc=NaN`；不能只保留进球率而省略这些限制。

同种子独立重跑的 `episodes.csv` 完全一致：
`sha256:b87ba798638d4e39afd25ec2060e2566665277591c844e1023541f92d3308cdd`。
这证明逐局输出可重复，不等于本程序保存并验证了所有逐物理步状态数组。

## 使用限制

官方 README 明确 CC BY-NC 4.0，并将商业产品宣传 demo 列为禁止用途。
本次仅研究验证；任何宣传素材使用、再分发、商业集成都需要另行解决许可，不能因有公开 checkpoint 就默认允许。

## 固定源码的训练目标审计

检查 `tracking_env_cfg.py`、`config/g1/soccer_flat_env_cfg.py` 与
`mdp/rewards.py`，没有导入或启动硬件、Isaac Lab 训练或更改参考代码。

基础 tracking 的姿态、身体位置、线速度、角速度项权重各为 1；
action-rate 为 -0.1，关节限位为 -10，非期望接触为 -0.1。
足球 proximity 子类将全局 anchor 位置跟踪设为 0，加入目标接近度 1、
腰部 action-rate -0.25、骨盆方向 -1、足部位置跟踪 1。
这体现了“保持运动先验，但释放足球任务需要的位置约束”，不是废除所有跟踪约束。

Kick 子类新增：预期脚首次接触 50、侧向摆脚对齐 50、球速方向对齐 30、
球速 10；垂直球速惩罚权重是 -0.0，不能说该配置启用了这一惩罚。
接触检测的水平力阈值为 10 N。方向和速度奖励只在正确脚接触后的
5 个 reward 调用窗口内激活；球速项为随水平速度增大而增加的饱和函数。

因此它是射门专家目标，**不能照搬成接球卸力目标**。
尤其不能把球速奖励作为 ABSORB 阶段奖励。本项目仍保持原判决标准，
未因参考审计而开新的 reward 参数搜索。

Moving 子类的训练初始球速按 x、y 各 [-0.3, 0.3] m/s 采样、z=0；
它与上面的 MuJoCo smoke 使用的速度模长区间不是同一个随机化定义。
源码还有一个需注意的配置细节：足球子类的一段 orientation 配置赋给
`self.motion_body_ori` 而非 `self.rewards.motion_body_ori`，不能据此宣称覆盖了
基础 reward 中的 body 列表。以上是源码审计，不是运行时配置导出证明。
