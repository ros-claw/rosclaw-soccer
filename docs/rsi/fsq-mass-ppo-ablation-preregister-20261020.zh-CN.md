# FSQ 实际执行符号的概率质量：有界 PPO 消融

## 要检验的机制

当前高斯先采样 64 维连续 raw，经原 SONIC tanh/round 变成 FSQ32
运动 token，肌肉解码器只消费 token。两个不同 raw 样本可能得到
完全相同的 token，之后实际关节目标和物理动作也相同。

原始 PPO 用 raw 的连续密度计算 ratio。这不是错误算法，优化潜变量
策略也有合法的策略梯度；但同一实际动作内的无效 raw 差异会影响
梯度。新对照假设：把每个实际符号对应的整段高斯质量积分，能减少
这类干扰。不是保证改善，也不把两个旧候选失败归咎于唯一根因。

## 数学与数值合同

原量化式保持：

`k = round(15.515501022338867 * tanh(raw + .03223699703812599) - .5)`

`token = k/16, k ∈ {-16,…,15}`。

内部 bin 的 raw 边界为
`lo=atanh(k/a)-s`、`hi=atanh((k+1)/a)-s`，最小/最大端 bin 分别
延伸至负/正无穷。概率质量是 `Φ((hi-μ)/σ)-Φ((lo-μ)/σ)`，64 维
条件独立符号的 log mass 相加得到实际运动向量的 log-prob。

使用 float64 的 log-CDF 和稳定 log-difference，上尾用对称形式，
避免 CDF 接近 1 时相减丢精度；端 bin 用有限占位再显式处理，避免
反向传播 `0*inf`。参考 [PyTorch 官方 log_ndtr 定义](https://docs.pytorch.org/docs/stable/special.html#torch.special.log_ndtr)，
并核对本地 Torch 2.13.0 的实际 API，而非照搬网络文档版本。
这是理想连续高斯的 bin 模型，有限精度 RNG/舍入仍是数值近似；
没有修改实际采样器或给 sampler 换精度。

## 严格配对

固定起点 seed29 BC 和原来的完整 32 个 on-policy 回合，不丢弃失败；
仍冻结肌肉/编码器/GRU，保持 4 次 Adam1e-4 更新、原逐帧奖励、
同一 GAE 和考试。与首次 1/40 候选比较时，唯一干预是 PPO ratio
从 raw 密度换成已执行 FSQ token 的概率质量；不叠加接触回分配。

old log mass 从已核验的原采样策略重新积分，同一个已记录符号；
不是误用原 raw log-density，也不把新策略当作 old。额外核验实际
记录 raw 用原 Torch 运算确实产生了物理执行的 token。
先验 KL 仍是连续高斯 KL（对量化后 KL 的上界），entropy 正则仍
为原连续高斯 entropy，不将它谎称离散 entropy。

保留所有原数据，新版本只变 learner/math，所有共同 actor、muscle、
physics、rollout 源码须逐字节一致。输出新候选、新 receipt，完整
40 课程及独立重放，绝不改原始成绩或自动晋升。

## 预检与可推翻的预测

- 五组均值覆盖中间与端点，32 个 bin 质量归一化，全部尾部梯度有限。
- 100,000 个原 sampler 样本的符号直方图与积分概率在声明容差内一致。
- 原 32 回合采样概率全量预检通过，joint log mass 范围约 -57.08 到
  -14.71；这只是概率计算检查，不是新仿真或能力提升。
- raw 与 mass 两种 PPO 都必须给 actor/critic 有限非零梯度，冻结模块
  不得变化；源在任何真实模型训练/物理分配前冻结。

若减少无效 raw 差异确有帮助，应看到原考试新增成功而保留旧成功，
并在后续独立种子/Fresh 中成立。若仍无净增长，停止把量化似然当作
充分解释，转向 own-state 接触教师与 motion64 DAgger、策略表示和
可达性研究，而非无限 reward/lr sweep。不得用 loss 降低代替验收。
