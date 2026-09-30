# v306 独立复核：训练结果必须能从原始物理重新计算

## 范围与当前状态

运行源固定在 `5b1f27c7c97a503c4759e54aa89b9fb260386a95`，目录
`/code/rosclaw/rsi-source-v306`。本次新增复核代码在另一个工作树开发，
不改变正在运行的策略、课程、排序或仿真源码。

前代在 12 个课程上的身体与球轨迹已逐字节重现。新 37 参数模型的
初始副本也重现了前代物理。计划 408 次物理回合，仍仅有 12 个不同
已消耗课程；408 不是 408 个独立测试场景。

第一次独立进度复核：5 个候选、60 次候选物理回合。最优仍是前代
初值：7/12 高质量，保留七个前代成功，但三个旧干净脚触球与一个旧
高质量仍退步。**此时训练未完成，开发门未通过，更没有球队突破。**
该快照不会在后续训练更新时被覆盖：
`/code/rosclaw/rsi-failure-curriculum-motor-v306-review-progress1.json`
`sha256:5c36cd71b7f605ff6aef1bec95b6ef34735b164b8c37d594fb09bd889f3b4e1d`。

## 新增只读复核能力

`failure_curriculum_evidence.py` 与 CLI `rsi_review_failure_curriculum.py`：

- 每个候选必须有完整 12 课程，拒绝用部分课程报告“通过”。
- 验证模型、训练 commitment、仿真源码、身体资产、课程与父报告绑定。
- 重建导航反馈、冻结摆脚与学习残差到实际关节目标的动作证据。
- 从球轨迹、接触记录、身体高度重新计算方向、位移、碰撞与奖励。
- 独立计算并命名五种保留损失，输出具体退步课程；奖励不能覆盖遗忘。
- 初始模型逐一检查前代、新源重现、初始新模型三者的身体和球轨迹。
- 已输出的训练候选成绩与独立重算不一致即拒绝；最终汇总还必须有
  全部 32 个候选，并与独立最佳排序一致。
- 未完成训练即使临时达到 12/12，也不能标记整个开发实验通过。
- 新留出、CPU MuJoCo、连续球队资格始终明确为未完成；不写注册表，
  不自动打开测试集，不授权硬件。

复核输出只接受新文件，不覆盖原报告、不写训练目录，不启动物理仿真。
原始失败轨迹和旧模型全部保留。

## 运行

在 Soccer 项目根目录，使用已验证 Python 环境：

```bash
PYTHONPATH=.:src:scripts:/code/rosclaw/rosclaw_core_20261001_readonly/src \
  /code/rosclaw/rosclaw_soccer_s2/.venv/bin/python \
  scripts/rsi_review_failure_curriculum.py \
  --training-root /code/rosclaw/rsi-failure-curriculum-motor-v306 \
  --validation-root /code/rosclaw/rsi-bilateral-motor-validation-v305 \
  --output /code/rosclaw/rsi-failure-curriculum-motor-v306-review-NEW.json
```

训练期间可生成只读阶段快照；最终应对完成汇总再次生成新复核文件。
模型仍是 CEM 学习的关节残差基元，不宣称神经 actor-critic/PPO 或
直接关节力矩学习。本阶段专注触球盆地，不能替代传球、射门、扑救、
自主 4v4 的完整验收。
