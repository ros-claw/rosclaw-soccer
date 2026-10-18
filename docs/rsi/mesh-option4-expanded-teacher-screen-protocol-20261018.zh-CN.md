# 已启动：option4 的跨状态、球初态扰动与保持筛查

## 为什么现在做这项，而不是继续PPO

三seed真实力矩学习已被完整审计否定为无净增长，不能重复同族调参。
另一方面，固定catalog的option4已在raw8/56各自安全救回原失败。
先检查这个已有动态有效控制能否扩展，比继续优化未证明必要的新IK
或扩大弱教师上的神经训练更有依据。

option4由**已消费的公开前缀**选出，选择有偏且已披露，不是盲测。
稍后option6也在两状态通过，但本研究不因此更换冻结候选或调参数。
原catalog的其余全部结果仍须完成，终态审计不可省略。

## 固定执行与复用

| 内容 | 新执行端点 | 已有端点复用 |
|---|---:|---:|
| 8个固定派生公共课程：None与option4配对 | 16 | 0 |
| 原raw4/32成功的option4保持检查 | 2 | None各1 |
| 原raw8/56失败的option4救回 | 0 | None/option4各2 |
| 合计 | 18 | 6 |

共12个不同课程、24个配对端点；不是24个独立场景，也不把300帧当成
300条独立任务回报。8个派生课程来自既有planning_contexts(plan)，
source_group固定64/65/68/69/72/73/76/77，四个x/速度符号类别各2个。
每个类别两个不同source draw再按原有规则扰动：x增加.01/.02m，
y乘.9/.8，vx乘.95/1.05。所有课程实际通过validate_context_source，
没有裁剪边界、重抽球位或选择成功轨迹。

扰动仅指这些球初态变化，**不包括**新质量、摩擦、延迟、视觉噪声、
外力、多人对抗或高球。四个旧raw与派生课程均已公开/消费，非Fresh。
控制和候选使用同一课程、父模型与motor采样种子，Foundation/PD保持；
None也不是“没有Foundation或没有旧motor”，只是没有新增B/C控制。

这12课程是有预算的教师可用性筛查，不替代总纲的四类各4初态pilot、
40场Expanded、旧52开发与隔离Fresh、历史保持及最终球队闭环要求。
通过本轮只能决定下一研究投入，不能宣称已经完成G3–G6或球队自进化。

## 已实际冻结并启动

- 冻结生产源：`8d748da6dcd0badfe75676e06a01be2851efb997`。
- 执行入口：`scripts/rsi_mesh_option4_teacher_screen.py`。
- 外置配置：`rsi-mesh-option4-teacher-screen-config-20261018.json`。
- 外置root：`rsi-mesh-option4-teacher-screen-20261018`。
- 启动清单：`rsi-mesh-option4-teacher-screen-launch-pins-20261018.json`。
- 用户级服务：`rosclaw-mesh-option4-teacher-screen-20261018.service`。
- 15:29:14实际启动PID2077170；15:29:39独立reference分配，
  15:36:23进入固定mean分配。记录时尚无新body成绩，不能把初始化算训练。

正式预检输出
`OPTION4_SCREEN_18_JOBS_12_PAIRED_COURSES_6_REUSED_ZERO_MODEL_ALLOCATIONS`。
先验证六个旧端点的文件字节、primitive标签和既有独立资格组合，
然后才允许分配模型。启动前验证56GiB MemAvailable、冻结HEAD/文件和
旧44体服务真实正常终态；当时仅旧catalog占另一完整模型槽位。

固定deadline1792316400（17:40），RuntimeMax2h、Restart=no；最大18次
新原生身体与18次独立CPU重演，54000+54000子步、零PPO/新优化更新。
内存High8G/Max12G/Swap4G、CPU150%、CUDA空。初始化ONNX的DRM发现
警告不作为GPU测试或通过证据；本实验明确使用CPU provider。

所有失败和删失保留，数据篡改/非有限值/合同不一致才视为致命故障。
一次性launch marker防止源/入口失败后无限重试；不因等待延长deadline。
生产源和配置已冻结，不修改已运行的代码。

## 独立终态闭环已经排队

另一个冻结审计源：`5329aa6ddfd2eab00e7d4d68160166891c834723`。
入口 `rsi_read_mesh_contact_native_pilot.py --profile teacher-eighteen`。
外置输出：`rsi-mesh-option4-teacher-independent-readback-20261018.json`。

用户级timer `rosclaw-mesh-option4-teacher-audit-20261018.timer` 已启用。
仅当本18体完整终态、上游16体完整审计实际成功退出、结果文件存在且
无既往start marker时启动；已失败的审计不静默自动重试。

审计会验证18份新完整原始证据、账本及预先intent、精确课程/采样种子、
planning source view、option/执行模式、原始MJB世界、CPU/Foundation和
完整encoder消费证明，再绑定原四例终态及上游完整catalog审计。
六个复用端点必须恰好对应原收据，并重算完整12课程rescue/regression/
newly-unsafe矩阵。复用依靠既有完整终态读回，不声称这里又跑了六次物理。
新生产者保存完整prepared_pins/source_heads，审计严格比较，不走旧缺口。

审计及计划/几何相关107项回归已实际通过，其中新18体合成完整链和
重封印篡改9项；不计为107次仿真成功。观察器已切换为真实两个活跃
研究，而不是继续观察已经结束的三seed worker。

## 结果如何改变下一步

- 旧能力或安全退化：不认可无条件全局option4教师，先研究适用域与
  失败状态；不挑好的子集冒充通用成功，也不重启旧PPO族。
- 没有跨状态有效性：回到接触速度/相位/姿态证据；表面点原型仅为
  后备可检验机制，不因为它的静态误差小就自动切换。
- 有真实配对收益且保持/安全成立：扩大教师状态及扰动覆盖，再开展
  接触条件化学生/DAgger与独立种子闭环学习；不是训练固定option编号
  后把旧motor没更新的选择网络称为新小脑。

所有teacher_qualified、promotion、hardware和Fresh授权仍为false。
整个球队与连续比赛宣传视频目标仍active，未完成。
