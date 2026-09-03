# Microduck 单腿训练实验账本

## 记录范围

本文记录 2026-09-02 至 2026-09-03 围绕 Microduck 单腿站立、单腿跳、
落地恢复和下蹲几何进行的训练探索。

这不是只保存最终方案的总结，而是保存完整决策链：

- 当时要解决什么问题；
- 为什么认为某个改动可能有效；
- 从哪个 checkpoint 继续；
- 实际改变了什么；
- 观察到了什么；
- 为什么继续、回滚或淘汰；
- 哪些结论后来被证明错误。

事实来源按可信度排序：

1. 本地保存的 checkpoint、`params/*.yaml`、TensorBoard event 和评估 JSON；
2. Vast 训练日志与监控状态；
3. Git commit 与当前工作区补丁；
4. 当日操作会话记录；
5. 无法从制品重新验证的内容会明确标记为“会话重建”。

## 当前状态

记录时间：2026-09-03。

```text
Vast instance: 49723497
状态: stopped / exited
GPU 费用: $0/hour
磁盘保留费: $0.0277777778/hour
原运行总价: $0.3677777778/hour
GPU: RTX 4090
最新完整 checkpoint: model_525.pt
付费训练: 已暂停
```

完整停机归档：

```text
/Users/bingo/Documents/Codex/2026-09-02/wo-x/work/cloud/paused-49723497-20260903/
```

归档包含：

- `model_0.pt` 至 `model_525.pt`；
- TensorBoard event；
- ONNX；
- `params/agent.yaml`；
- `params/env.yaml`；
- 完整训练日志；
- 未提交源码补丁；
- `SHA256SUMS` 和校验结果；
- Vast 停机状态 JSON。

当前本地姿态实验室：

```text
http://127.0.0.1:8080
```

当前尚未批准恢复训练。正在讨论的新目标是：

```text
单腿站立 -> 单腿下蹲 -> 恢复原单腿站立
```

## 目标如何演变

### 最初目标

训练一个左右可命令的单腿站立策略。

### 第二阶段目标

从单腿站立模型出发，学习重复单腿 hop。

### 第三阶段目标

统一四种用户语义：

```text
左腿站
右腿站
左腿跳
右腿跳
```

### 第四阶段目标

把 hop 变成一次完整动作：

```text
准备 -> 压缩 -> 伸展 -> 离地 -> 同脚落地 -> 恢复
```

### 当前重新定义

暂时移除离地、空中和落地，只研究：

```text
单腿稳定站立 -> 支撑腿正确折叠 -> 回到原单腿站立
```

原因：如果单腿支撑下不能完成可控的深蹲与恢复，继续训练单腿跳没有物理基础。

## 关键指标定义

后续所有实验必须区分两类指标。

### 在线训练分布指标

用于立即发现平坦或退化趋势：

- 左右起跳率；
- 左右落地率；
- 左右完整完成率；
- 左右平均高度；
- 左右下蹲深度；
- 左右支撑膝屈曲；
- 失败率；
- `nan_state`。

这些指标可以直接淘汰明显无效的 Trainer。

### 固定确定性评估

用于 checkpoint 晋级：

- 固定 standing start；
- 固定 command side；
- 同时固定 reset side；
- deterministic actor；
- 每侧固定 episode 数；
- 起跳、同脚落地、恢复完成、平均/中位高度。

固定评估不是拒绝坏趋势的前置条件。

## 实验时间线

### A. 单腿站立基础

| ID | 时间/Run | 背景与假设 | 结果 | 裁决 |
|---|---|---|---|---|
| A00 | 2026-09-02 12:47–12:57，多次 5/20 iteration smoke | 验证新单腿任务能构建、61D 观测和左右 command 可工作 | smoke 可运行；默认 PPO 参数仍为 `lr=1e-3`、`std=1.0` | 进入正式训练 |
| A01 | `single-leg-8192-20260902` | 8192 环境并行训练左右单腿站立 | 建立基础行为，但严格成功判定仍需继续收紧 | 继续 strict 版本 |
| A02 | `strict-cloud-smoke`，约 iter 2326 | 用更严格的持续单腿状态筛掉短暂抬脚 | 能继续学习，但奖励/终止仍可被利用 | 迭代 strict-v2 |
| A03 | `strict-v2-single-leg`，iter 2509 | 增强有效 hold 与失败成本 | 静态保持增强；动态探索能力下降 | 继续简化 |
| A04 | `strict-v3-single-leg`，iter 3263 | 继续提高真正单腿保持稳定性 | 单腿站立更稳定，但模型更偏静态局部最优 | 继续 minimal |
| A05 | `strict-v4-minimal`，checkpoint 3750 | 删除多余 shaping，只留下最小严格合同 | 得到后续 hop 的基础单腿站 actor | 接受为 stand 起点 |

#### 单腿站立为何能学出来

核心不是一次性“成功/失败”，而是有连续梯度：

- command 随机指定左右支撑腿；
- 重心目标在 1.5 s 内从双脚中点移动到支撑脚；
- 重心更靠近支撑脚就获得更高回报；
- 支撑脚接触、少滑动有利；
- 摆动脚逐渐抬高有利；
- 摆动脚触地、身体倾倒、非脚接触不利。

随机探索中任何微小正确动作都会比完全错误动作得分稍高，PPO 逐步放大这种倾向。

#### 负面教训

只训练静态单腿站会形成过拟合：

- 动态动作被视为破坏平衡；
- action-rate 等正则鼓励不动；
- 探索收缩；
- 后续下蹲/跳跃很难离开静态局部最优。

因此后续不应重新从头训练单腿站，而应把 A05 作为需要保留的旧能力。

### B. 重复 Hop 与动作发现

| ID | Run | 假设 | 结果 | 裁决 |
|---|---|---|---|---|
| B01 | `single-leg-hop-8192`，iter 4060/7750 | 从 stand3750 直接学习重复 hop | 长训练产生动作，但没有可靠固定评估证明完整跳跃链 | 淘汰为最终方案 |
| B02 | `command-conditioned-hop`，iter 238 | 给 stand/hop 增加 command 区分 | command 语义可建立，但动作质量未验证 | 转向自由节奏 |
| B03 | `free-tempo-hop`，iter 345，checkpoint 250 | 去掉固定 phase，让策略自己找节奏 | 奖励可上升，但不能证明真实离地/恢复 | 转向动作引导 |
| B04 | `guided-action-hop`，iter 98 | 用结构化动作块引导深蹲/伸展 | 方向可发现，但过强引导与旧站姿冲突 | 调低引导 |
| B05 | `guided-hop-005`，iter 62 | 将动作引导降低到 0.005 | 仍未形成可靠完整链 | 转向行为克隆 |
| B06 | `bc-landing`，训练到约 iter 266 后 traceback | 先用 BC actor，再用 PPO 学落地 | reward 上升但训练异常结束，且未形成可靠固定验收 | 淘汰 |
| B07 | `hop-stage-a`，250 iter | 只训练早期动作片段 | mean reward 可到约 20，但总奖励不是 hop 成功 | 仅保留为探索证据 |
| B08 | `true-hop-stage-a`，250 iter | 用更严格真实离地定义替代代理动作 | reward 可到约 23，仍缺少完整落地恢复证明 | 继续 BC |
| B09 | `bc-v2-imitation`，250 iter | 更直接地模仿候选动作块 | 可稳定训练，成为 v6 构建的一部分 | 合并进入 v6 |
| B10 | `bc-v2-landing-shape`，250 iter | 在 BC 基础上增加落地姿态塑形 | 没有独立晋级证据 | 不单独采用 |

#### 这一阶段的共同问题

- reward 增长经常来自代理 shaping，而非真实离地；
- repeated hop 与用户的一次性动作语义不一致；
- 自由节奏减少人为时序约束，但稀疏任务难以探索；
- 固定动作块容易产生“照着做”，但不一定能反馈纠错；
- 单腿站模型本身偏静态，PPO 很难主动破坏平衡去探索跳跃。

### C. Robust Block BC v6

制品：

```text
/Users/bingo/Documents/Codex/2026-09-02/wo-x/work/cloud/milestones/
2026-09-03-robust-hop-bc-v6/
```

v6 command：

```text
左站: [0, -1, 0]
右站: [0, +1, 0]
左跳: [1, -1, phase]
右跳: [1, +1, phase]
```

内部 phase：

```text
压缩: -1
伸展: +1
稳定: 0
```

时序：

```text
准备: 1.5 s
压缩: 0.22 s
伸展: 0.12 s
```

v6 是从 stand actor 与人工筛选动作块构建的行为克隆 actor，不是纯 PPO 从零发现。

#### v6 原始验收

10 秒、每 command 1024 环境：

- 左站存活 `99.6%`；
- 右站存活 `99.9%`；
- 左 hop：272 起跳、237 完成；
- 右 hop：86 起跳、86 完成。

60 秒、每 command 256 环境：

- 左 hop：63 起跳、58 完成；
- 右 hop：37 起跳、37 完成。

这些结果来自旧的行为验收定义，不能直接与后来更严格的 standing-start evaluator 混为一谈。

#### 严格固定评估基线

每侧 128 局：

| checkpoint | 左：起跳/落地/完成 | 右：起跳/落地/完成 |
|---|---:|---:|
| accepted v6 | 29/28/4 | 7/3/2 |

v6 的价值：

- 已包含四种 command 语义；
- 有真实起跳样本；
- 有少量完整恢复；
- 是后续所有保守 PPO 的父 actor。

v6 的限制：

- 左右明显不平衡；
- cross-side transition 很差；
- 压缩动作浅；
- 行为高度依赖人工动作块；
- 严格 standing-start 完成率仍低。

### D. 纯 PPO、锁 std 与高度/恢复尝试

Vast 实例：`49723497`。

| ID | Run/策略 | 假设 | 固定评估/在线结果 | 裁决 |
|---|---|---|---|---|
| D01 | 默认 PPO | 直接从 v6 继续，PPO 会提高完整率 | model250 左右均 0/0/0；model500 仅右 3/3/0 | 严重退化 |
| D02 | locked std model250 | 锁定 v6 的 `std=0.005` 可防止遗忘 | 左右仍 0/0/0 | 仅锁 std 不够 |
| D03 | ultra-conservative model250 | `lr=1e-6`、1 epoch、std 锁定 | 左 39/38/3，右 23/18/2 | 起跳/落地增强，恢复尚在 |
| D04 | ultra-conservative model500 | 继续同一路线会巩固能力 | 左 38/37/0，右 23/14/0 | 恢复被逐步擦除 |
| D05 | height-off continuation | 过早优化高度可能压制恢复；先关闭高度 | 未留下可晋级的 completion 改善 | 未证明 |
| D06 | recovery-first | 提高恢复进度奖励，先保住落地后站立 | 在线训练指标上涨，但严格效果未形成晋级证据 | 未证明 |
| D07 | uncapped-height-recovery | 成功后高度每 10 mm 线性奖励且不封顶 | 见下表 | 起跳保持，恢复继续退化 |

#### uncapped-height-recovery 严格评估

| checkpoint | 左：起跳/落地/完成 | 右：起跳/落地/完成 | 总完成 |
|---|---:|---:|---:|
| model250 | 40/39/1 | 24/14/3 | 4/256 |
| model500 | 37/37/1 | 25/21/1 | 2/256 |
| model750 | 40/37/0 | 27/20/1 | 1/256 |
| model1000 | 34/30/0 | 20/14/2 | 2/256 |

结论：

- 纯 PPO 可以增加起跳或落地；
- 但确定性恢复能力持续消失；
- 总 reward 或训练 completion 不能证明完整任务进步。

### E. 教师动作锚定

根因假设：

```text
PPO 均值策略逐步偏离 v6，擦除原有恢复动作。
```

实现：

- 保存冻结 v6 teacher；
- PPO 在 phase=0 的准备/恢复阶段对 teacher action mean 做 MSE 锚定；
- 压缩和伸展 phase 不锚定；
- teacher 随 checkpoint 保存；
- std 固定 `0.005`；
- `lr=1e-6`；
- 1 learning epoch。

#### 锚定 0.5

| checkpoint | 左：起跳/落地/完成 | 右：起跳/落地/完成 | 总完成 |
|---|---:|---:|---:|
| model0 | 29/29/4 | 16/11/6 | 10/256 |
| model100 | 33/33/0 | 17/11/3 | 3/256 |

结论：初始 checkpoint 明显好于 v6，但 100 iter 后仍遗忘恢复。

#### 锚定 5.0

| checkpoint | 左：起跳/落地/完成 | 右：起跳/落地/完成 | 总完成 |
|---|---:|---:|---:|
| model0 | 38/38/7 | 11/8/4 | 11/256 |
| model50 | 36/36/5 | 19/11/5 | 10/256 |
| model250 | 39/39/1 | 18/13/1 | 2/256 |

结论：

- model0 成为当日严格 evaluator 的最佳 checkpoint；
- 5.0 锚定只能延缓退化；
- 长期 PPO 仍把恢复动作擦除。

#### 锚定 50.0

假设：更强锚定可以长期保留 v6。

结果：没有完成独立固定晋级评估；在线 teacher drift 仍增长。随后暴露更严重的
任务定义和失败终止问题，实验被替换。

### F. 状态机、时间、奖励与下蹲几何修正

| ID | 改动 | 背景 | 结果 | 裁决 |
|---|---|---|---|---|
| F01 | 删除压缩开始时的硬失败 | 模型刚开始下蹲就被 reset；系统用 command 时钟预测未来失败 | P0 逻辑修复 | 永久保留 |
| F02 | 删除伸展 phase 结束时的硬失败 | 仍有物理动量的动作被固定时钟掐断 | P0 逻辑修复 | 永久保留 |
| F03 | 查看器恢复真实回合 | 旧 viewer 强制 3600 s 且删除成功终止，跳完无限站立 | 默认 viewer 改为真实 episode；diagnostic 才无限 | 永久保留 |
| F04 | episode 6 s -> 3 s；prepare 1.5 s -> 0.8 s | 失败尝试与准备站立占用过多样本 | 单位步数动作尝试增加 | 未完成质量验证 |
| F05 | completion20 + height1 -> height20 | 最低 3 mm 成功几乎与更高跳同价 | 强化成功后的高度差异 | 在线高度仍未改善 |
| F06 | reset 35/15/50 -> 60/25/15 | 50% airborne 让训练 completion 主要反映恢复，而非完整动作 | 增加 standing-start 数据 | 在线趋势仍平 |
| F07 | airborne reset 不领取历史高度 | 空中重置可用预存峰值领取高度奖励 | 修复免费高度 | 保留 |
| F08 | 压缩目标 10 mm、weight 0.5 -> 20 mm、weight 5 | 视觉上几乎没有下蹲 | 发现 curriculum 又把 weight 改回 0.5 | 当前 run 立即无效 |
| F09 | 修正 curriculum 的压缩 weight | 配置本体与课程冲突 | 正确应用 weight5 | 很快被“奖励封顶”问题替换 |
| F10 | 下蹲深度奖励线性不封顶、成功后兑现 | 固定 20 mm 仍是 pay cap；失败下蹲不应获利 | 修复封顶和失败 farming | 仍只看高度 |
| F11 | 加入支撑膝屈曲 | 模型通过髋/踝/整体下沉骗取高度奖励 | 下蹲深度与膝角开始上升 | 同步几何仍未验证 |
| F12 | 姿态实验室 | 训练前没有人工确认什么是正确下蹲 | 可直接调 14 关节、对称锁、预设和保存 | 当前本地研究工具 |

### G. 停机前最后一轮：knee-crouch-composite

Run：

```text
2026-09-03_13-07-46_knee-crouch-composite-uncapped-height20-fast3s-8192-20260903
```

父 checkpoint：

```text
teacher-anchor5-resume-model0 / model_0.pt
```

主要配置：

- 8192 envs；
- 3 s episode；
- prepare 0.8 s；
- crouch 0.35 s；
- extend 0.12 s；
- standing/compressed/airborne = 60/25/15；
- completion weight 20；
- successful height weight 20，10 mm = 1 unit，不封顶；
- successful compression weight 5，不封顶；
- teacher anchor 50；
- std 0.005；
- `lr=1e-6`；
- 1 epoch。

最后保存：

```text
model_525.pt
TensorBoard final step: 544
```

#### 在线训练分布首末值

| 指标 | step0 | step544 | 趋势 |
|---|---:|---:|---|
| 左完成率 | 0% | 4.65% | 上升但低 |
| 右完成率 | 0% | 1.63% | 上升但更低 |
| 左下蹲深度 | 0.86 mm | 7.17 mm | 上升 |
| 右下蹲深度 | 1.37 mm | 8.46 mm | 上升 |
| 左膝屈曲 | 6.33° | 12.39° | 上升但不足 |
| 右膝屈曲 | 6.55° | 9.60° | 上升但不足 |
| 左平均跳高 | 23.35 mm | 3.20 mm | 大幅下降 |
| 右平均跳高 | 25.21 mm | 1.87 mm | 大幅下降 |

step0 的高度受到 reverse-reset 初始分布影响，不能作为 standing-start 晋级基线。

最终裁决：

- 模型开始学习更明显的下降和膝关节运动；
- 没有证明形成正确髋膝踝折叠；
- 没有证明跳跃变高；
- 右侧明显弱于左侧；
- 用户要求暂停训练并重新定义目标；
- 实例在 model525 后停止并完整归档。

## 发现过的错误与后果

### 1. 把训练 reward 当成任务成功

后果：起跳/落地指标上涨时，严格完成率可能已经归零。

规则：在线指标可淘汰；固定 standing-start evaluator 才能晋级。

### 2. 默认 PPO 参数破坏继承 actor

默认 std 从 0.005 膨胀到约 0.025–0.031，原确定性动作被随机探索擦除。

规则：继承 actor 必须明确审计 std、entropy、LR、optimizer。

### 3. 用 command 时钟提前判失败

后果：刚下蹲或仍在伸展的动作被立即 reset，没有机会完成。

规则：时钟只能切 phase，不能预测物理动作必然失败。

### 4. 高度奖励封顶

后果：策略只追求达到最低合格高度，没有继续跳高的动力。

规则：用户要求“越多越好”的目标默认线性不封顶。

### 5. 下蹲只看根部高度

后果：模型可用髋前伸、踝折叠或整体倾倒降低高度，不需要正确屈膝。

规则：同一时刻检查支撑腿缩短、膝屈曲、根部下降、支撑和姿态。

### 6. 把不同时间的最大值拼成一次成功

后果：某一帧短暂弯膝，另一帧身体下降，也可能被计为正确下蹲。

规则：几何 composite 必须在同一 physics state 上计算。

### 7. reverse curriculum 污染直观趋势

后果：airborne reset 的初始高度让 step0 看似跳得很高；训练 completion
也可能主要来自恢复样本。

规则：每个指标必须按 reset kind 与 standing-start 分开。

### 8. 静态站立过拟合

后果：策略越来越不愿意产生动态动作。

规则：从已有 stand actor warm-start；新 command phase 从第一天就非零；只在站立
phase 保留 teacher/replay。

### 9. Viewer 与真实训练不一致

后果：无限 episode、删除终止或展示旧 checkpoint，让人眼观察失真。

规则：每次汇报必须记录当前展示 checkpoint、SHA、状态和 viewer URL。

### 10. 已知无效 Trainer 继续付费运行

后果：GPU utilization 正常，但 time-to-quality 为零。

规则：TensorBoard 已证明平坦/退化即可立即拒绝；没有替代方案时回滚最后有效合同。

## 当前新方向：单腿下蹲并恢复

当前只批准讨论，尚未启动训练。

目标：

```text
单腿站立 -> 支撑腿下蹲 -> 恢复原单腿站立
```

明确移除：

- 离地奖励；
- 起跳奖励；
- 跳高奖励；
- 空中 reset；
- 落地奖励；
- 速度目标；
- 速度限制。

建议初始数据：

```text
100% 已验证单腿站立状态
```

只有在“会蹲但不会恢复”被实际测量后，才增加少量 crouched reset。

建议 phase：

```text
0.0–0.5 s: 初始单腿保持
0.5–2.5 s: 下蹲
2.5–4.5 s: 恢复
4.5–5.0 s: 最终保持
```

不奖励速度。phase 只定义当前目标。

建议下蹲进度：

```text
crouch_score =
min(
  root_drop / 10 mm,
  support_leg_shortening / 10 mm,
  signed_support_knee_flexion / 15 deg
)
```

不封顶，只支付新的进展：

```text
reward_down = max(crouch_score_now - best_crouch_score_so_far, 0)
```

建议恢复进度：

```text
reward_return =
previous_distance_to_initial_stand - current_distance_to_initial_stand
```

最终连续保持原单腿站姿 0.5 s 后支付一次 completion。

离地：

- 不奖励；
- 不惩罚；
- 离地帧不计入下蹲或恢复进度。

## 每次训练的强制记录模板

以下模板必须在启动前填写 `PLANNED` 部分，并在停止或切换时填写 `RESULT`。

```markdown
### EXP-YYYYMMDD-NN — <run name>

Status: PLANNED | INITIALIZING | ACTIVE | PROMOTED | REJECTED | PAUSED | FAILED

#### Context

- 当前问题：
- 上一 run 的证据：
- 为什么现在必须改：

#### Hypothesis

- 单一可证伪假设：
- 预期改善的主指标：
- 可能恶化的保护指标：

#### Parent

- checkpoint：
- checkpoint SHA-256：
- actor/critic/optimizer/normalizer 是否完整继承：
- git commit：
- dirty patch：

#### Contract

- task id：
- exact command：
- environments：
- episode/phase timing：
- command distribution：
- reset distribution：
- reward table：
- termination table：
- std / entropy / LR / epochs：

#### Gates

- 在线拒绝 gate：
- 固定晋级 gate：
- 最晚决策 iteration：
- 最大预算：
- rollback checkpoint：

#### Operations

- Vast instance / offer / price：
- Trainer PID / PGID：
- remote run：
- local mirror：
- TensorBoard：
- viewer checkpoint / SHA / URL：

#### Result

- 实际 iteration / wall time / cost：
- 在线趋势：
- 固定评估：
- 视频/人工观察：
- 异常：

#### Verdict

- PROMOTED / REJECTED / PAUSED / FAILED：
- 原因：
- 保留的结论：
- 下一实验只改变什么：
```

## 维护规则

- 不覆盖旧结论；发现错误时追加 correction。
- reward 改动必须记录实际 weighted mass，不只记录 weight。
- phase、reset mix、checkpoint parent 和 optimizer resume 必须显式记录。
- 每个 run 必须保存精确命令，不接受“与上一轮类似”。
- 没有结果的 run 也必须记录为什么停止。
- 人眼观察必须描述具体动作，不写“效果不错”。
- 所有关键制品在销毁实例前生成 SHA-256。
- 当前账本是实验决策入口；运维细节继续记录在
  `docs/2026-09-03-vast-paid-gpu-training-operations.md`。

## 2026-09-04 预登记

### EXP-20260904-01 — single-leg-crouch-return-v1

Status: PLANNED

#### Context

- 单腿跳训练已暂停，Vast GPU 费用为 0。
- 当前要验证的更小物理问题是：已有单腿站 actor 能否在同一支撑腿上下降，再恢复原高度。
- 上一轮 knee-crouch-composite 同时包含跳跃、reverse reset、恢复和膝角复合，目标过多。
- 用户明确要求第一版只使用高度进度，不奖励具体屈膝、速度、COM 或倾斜。

#### Hypothesis

- 从 strict-v4 单腿站 actor warm-start；
- 100% standing start；
- 下蹲 phase 只奖励新的有效最低高度；
- 恢复 phase 只奖励新的有效最高高度；
- 最终回到 episode 初始高度 ±5 mm 并保持 0.5 s；
- 这个密集、无封顶、无速度偏好的合同足以学习单腿下蹲恢复。

#### Parent

```text
checkpoint:
/Users/bingo/Documents/Codex/2026-09-02/wo-x/work/cloud/
training-monitor-49645174-strict-v4/remote-training/model_3750.pt

SHA-256:
4f445e8eaa482b64596d69044730aafc42d3ed0bd78f52ce92e0cfa95d67a526
```

- actor/normalizer：继承；
- critic/optimizer/iteration：重新开始；
- 当前代码基线 commit：`1861ca0893b9ed77317757921b6085913913a5f1`；
- 当前旧 jump 实验 dirty patch 不得进入新任务提交。

#### Contract

- task id：`Mjlab-SingleLegCrouch-Flat-MicroDuck`；
- environments：smoke 64，正式 8192；
- episode：5.5 s；
- phase：0.5 s hold / 2.0 s down / 2.0 s return / 1.0 s final；
- command：左右各 50%；
- reset：100% standing；
- rewards：down5 / return5 / completion10；
- completion band：初始高度 ±5 mm，hold 0.5 s；
- 离地：0 reward、0 penalty；
- std：fixed 0.02；
- entropy：0；
- LR：`5e-5`；
- epochs：1。

#### Gates

- 在线拒绝：model25/50/100，任一侧连续不增长或 model50 completion=0 即拒绝；
- 固定晋级：每侧 128 局，model100 各至少 50 completion，平均 depth 至少 10 mm；
- rollback：strict-v4 model3750；
- 最大首轮预算：100 iterations gate，未晋级不自动续跑。

#### Operations

- Vast instance：49723497；
- 原价：`$0.3677777778/hour`；
- 当前状态：stopped/offline，`rentable=false`；
- 恢复超过 30 s 仍 scheduling 则不等待，转同价 4090 watcher；
- TensorBoard、自动评估、checkpoint 同步和 8080 viewer 必须在正式启动前 ready。

#### Result

- 尚未启动。

#### Verdict

- PLANNED；
- 执行文档：
  `docs/2026-09-04-single-leg-crouch-return-execution.md`。
