# OmniContact 接触数据：可用范围与字段审计

2026-09-20。来源为用户外置的 OmniContact 数据集；
[数据卡](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset/blob/main/README.md)。
CC BY-NC 4.0，仅研究，不把轨迹、派生模型或素材并入默认商业发布物。

本地足球部分实查 200 条唯一 capture、200 个不同文件哈希：
forward/left/right kick 各 30，carry 51，pickball-kick 59。
按原始 capture 的推荐划分为 train 130、val 27、test 43。
没有把同一轨迹的帧随机拆到不同集合，也没有把帧数称为独立试验次数。

检查了 29D joint_pos、39-body 四元数、物体位置、fps、四路二值 contact 的
形状、有限值和四元数范数，以及 capture 与 splits.csv 的关联。
所检查字段全部通过；不等于认证所有数组、每个身体索引或原始采集质量。
局部可选字段出现三种组合，不能强制假设每条数据都有 scale/table 字段。

一个明确的 schema 差异：README 列出的 base_quat_w 不存在于实查文件中，
实际提供 body_quat_w。检查程序改为验证真实字段，但**没有猜测哪一列是骨盆**。
身体及关节语义映射必须另行和官方 loader/URDF 对齐后，才允许构造动作 prior。

仅在训练集合枚举了 1,150 个足通道标签开启窗口；其中 66 个窗口在前后
0.2 s 的物体有限差分速度均值上满足“之前 >0.35、之后 <0.35 m/s”。
这不是 66 个接球成功：标签是注释，不是接触力；未确认每个足通道代表
脚—球而非其他接触，物体减速也可能来自其他作用。未读取 val/test 的窗口来选 prior。

风险与处理：

- 高：将射门、持球动作误标为来球接收。保留原任务类别，不生成虚假 receive 标签。
- 高：用二值标签代替物理脚球冲量。未来 teacher 仍需 CPU MuJoCo 验证。
- 中：字段与说明不一致。检查真实 shape、显式语义映射；缺失不补零。
- 中：连续时间帧高度相关。按 capture 分组，不把 1,150 个窗口视为独立数据源。

可复查外置证据：`receiving-mechanism-reboot/omnicontact-audit-v1.json` 和
`audit_omnicontact.py`，包含逐文件哈希、划分、字段清单和检查逻辑。
这是主实施中的定向数据质量检查；还没有训练 OmniContact prior 或通过 E2。
