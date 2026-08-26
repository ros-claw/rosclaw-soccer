# S86：G1 门将个人扑救表演集合

日期：2026-08-25
边界：`SIM_ONLY`；没有启动 ROS/DDS、串口、CAN 或真实机器人接口。
物理裁判：CPU MuJoCo 数值轨迹；视频像素不参与评分或晋级。

## 1. 交付结果

本轮生成了一个专门以门将为主体的 33 秒、1920×1080、30 fps H.264 合集：

```text
/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/
  s86-g1-goalkeeper-showcase-v2/
    rosclaw-g1-goalkeeper-showcase-1080p.mp4
    rosclaw-g1-goalkeeper-showcase-1080p.json
    contact-sheet.jpg
```

视频 SHA-256：

```text
sha256:c1e8aeebf803472e36fea37a44e2993953c999df8fa278cc34e9d725e3b13cba
```

主片包含 4 条已经严格双回放通过的高球扑救，每条分别使用门将近景、球门线或侧向
慢镜头呈现；片尾增加一条受控低位侧扑和一条连续双球扑救。展示镜头把前锋、传球者
移出画面，但门将和足球的冻结 qpos 没有改动，数值结论仍来自原物理实验。

## 2. 四个正式主镜头

主镜头来自 S85 三台 G1 同场实验中的 4 条独立轨迹。每条轨迹均满足：

- 第一台 G1 真实传球、第二台 G1 真实触球、门将手套真实接触同一个球；
- `NO GOAL`；
- CPU MuJoCo 第一次运行和严格重放的结果字典、轨迹摘要一致；
- 三角色稳定、关节/力矩边界通过、执行器零饱和；
- 标准 `7.32 × 2.44 m` 球门。

| 镜头 | 手套接触球心 y | 接触高度 z | 入射速度 | 结果 |
|---|---:|---:|---:|---|
| right-channel | +0.430 m | 1.409 m | 8.701 m/s | 手套扑出 |
| center-channel | -0.005 m | 1.459 m | 8.700 m/s | 手套扑出 |
| left-channel | -0.255 m | 1.459 m | 8.701 m/s | 手套扑出 |
| far-left-channel | -0.455 m | 1.459 m | 8.701 m/s | 手套扑出 |

四个接触点的横向总跨度为 `0.8850 m`。视频不是同一条轨迹横移复制，而是 4 条各自
有内容哈希、接触事件和安全结果的物理轨迹。

## 3. 受控侧扑片段

增强版从 S45 v190 的原生 MuJoCo-Warp 轨迹中只选择一个单回合通过样本：

- `world_index=102`；
- 目标 `y=-0.725 m, z=0.264 m`；
- `first_hand_save=true`；
- `qualified_save=true`；
- `stable_save=true`、`recovered=true`、`failed=false`；
- 最低骨盆 `0.5609 m`；
- 峰值根部角速度 `2.6243 rad/s`。

这是一条有明显低位侧向扑救和恢复的轨迹，但不是腾空飞身。其训练报告整体状态为
`REJECTED_NO_SAFE_CANDIDATE`，所以画面永久标注“THIS STABLE SAVE ONLY”，不能把一个
漂亮样本说成策略总体已经晋级。

## 4. 连续双球扑救片段

片尾选择 S37 CPU MuJoCo 回合 `seed=98824`：

- 首扑成功；
- 首扑后恢复成功；
- 第二扑成功，且第二扑由手套完成；
- 第二球释放时横向回位误差 `0.0149 m`；
- 最低骨盆 `0.7560 m`；
- 峰值根部角速度 `1.0161 rad/s`；
- 无失败和关节边界问题。

这是真实连续双球回合，不是把两条独立视频拼成二扑。但该候选在 64 回合整体考场中
因相对父策略的二扑提升和总回报不足而被拒绝，状态为
`REJECTED_BY_CPU_MUJOCO_EXAM`。因此这里只能声称“该单回合成功”，不能声称该策略
获得稳定的连续二扑能力。

## 5. 为什么没有加入“摔倒后二次扑出”

当前证据库有：

- 真实扑救后的倒地状态；
- 9/9 多姿态起身与 Capture→locomotion 交接可达性；
- 独立的连续双球成功回合。

但还没有一条同一连续物理剧集同时满足：

```text
首扑 → 真实倒地 → 自主起身/恢复 → 第二球释放 → 第二次手套扑救 → 两球均未进
```

因此本轮 manifest 固定写入：

```text
fall_then_second_save_included=false
fall_then_second_save_reason=NO_CONTINUOUS_STRICT_PHYSICS_EPISODE_AVAILABLE
```

视频剪辑不能替代这个缺失的完整技能链。下一步若开发该能力，必须在同一 MuJoCo
回合里设置有序事件门，并禁止 reset、teleport 或按结果选择起身路线。

## 6. 工程加固

新增 `goalkeeper_showcase_video.py`：

- 只接受 S85 全部门控通过且每条轨迹严格重放一致的主证据；
- 对轨迹、请求、源视频、源考试和最终视频逐文件绑定 SHA-256；
- 单独验证受控侧扑的手套扑救、合格扑救、稳定和恢复字段；
- 单独验证连续二扑的首扑、恢复、二扑、手套接触和稳定上限；
- 将两个旧候选的整体拒绝状态写入 manifest 和视频像素；
- manifest 自哈希，任一源文件或权限声明漂移均 fail-closed；
- 明确 `visualization_only=true`、`pixels_used_for_scoring=false`、
  `promotion_eligible=false`。

新增测试覆盖：

- manifest 和全部源文件的内容绑定；
- 篡改源文件时拒绝；
- 禁止把不存在的摔倒后二扑字段改成 `true`。

## 7. 验证结果

- 聚焦测试：`7 passed`；
- Ruff：通过；
- mypy：通过；
- compileall：通过；
- `git diff --check`：通过；
- 最终 manifest 重新加载并通过完整内容校验；
- 逐段抽帧复查：门将、球、标准球门、手套动作和标签可见，无黑帧或编码失败；
- 视频：约 11.29 MB，33.0 秒，990 个实际解码帧（容器索引显示 991，属于 concat
  边界的末帧索引差异）。

额外回归中，S80 的历史证据校验按设计因当前共享实现哈希与旧冻结实现不同而
fail-closed；这不影响 S86 所绑定的 S85 轨迹，但也说明旧证据不能冒充当前代码重新
晋级。

## 8. 诚实结论

当前可以说：

> ROSClaw 已生成一个门将专场，包含四个不同横向高球的严格手套扑救、一个单回合
> 受控低位侧扑与恢复，以及一个单回合连续双球扑救；所有展示均有物理状态和内容哈希
> 可追溯。

当前不能说：

- 已经实现真正腾空的飞身扑球；
- 已经实现摔倒后起身再扑第二球；
- 两个开发 bonus 所属的策略已整体晋级；
- 仿真结果已经授权部署到真实 G1。
