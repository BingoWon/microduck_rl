# 2026-09-04 单腿下蹲恢复执行文档

## Goal

在现有左右单腿站立 actor 基础上，训练一个更简单的单次动作：

```text
单腿站立 -> 支撑腿下蹲 -> 恢复原单腿站立高度
```

本 Goal 已在当前任务中显式创建。后续上下文即使压缩，也必须继续按本文执行，
不得恢复旧的单腿跳训练目标。

## 当前状态

```text
日期: 2026-09-04
Vast instance: 49723497
状态: offline / intended=stopped
GPU 费用: $0/hour
磁盘费: $0.0277777778/hour
原运行总价: $0.3677777778/hour
GPU: RTX 4090
rentable: false
```

原实例数据完整保留。恢复命令仍可调用，但 Vast 明确说明恢复取决于原机器当时的
资源可用性。

## 父 checkpoint

采用 strict-v4 单腿站立 checkpoint，而不是任意同名副本：

```text
/Users/bingo/Documents/Codex/2026-09-02/wo-x/work/cloud/
training-monitor-49645174-strict-v4/remote-training/model_3750.pt
```

```text
SHA-256:
4f445e8eaa482b64596d69044730aafc42d3ed0bd78f52ce92e0cfa95d67a526
```

checkpoint 内容：

- actor；
- critic；
- optimizer；
- iteration 3750；
- 61D observation normalizer；
- 14D action distribution。

新任务只 warm-start actor 与 observation normalizer。critic、optimizer 和 iteration
重新开始，因为 reward 与 episode contract 已改变。

## 明确非目标

第一轮不训练：

- 离地；
- 起跳；
- 腾空；
- 落地；
- 跳跃高度；
- 目标速度；
- 最大速度；
- 精确膝角；
- 精确髋角；
- 精确踝角；
- 重心跟踪；
- 身体倾斜 shaping；
- 动作平滑 shaping；
- 空中/压缩 reverse resets。

离地本身不奖励、不惩罚。离地期间的高度变化不计入任务进度。

## Command

一个 actor 同时支持左右两侧：

```text
[1, -1, phase] = 左腿支撑下蹲循环
[1, +1, phase] = 右腿支撑下蹲循环
```

内部 phase：

```text
0  = 初始保持 / 最终保持
-1 = 下蹲
+1 = 恢复
```

用户只选择左或右下蹲循环，不直接控制 phase。

## Episode 时序

```text
0.0–0.5 s  初始单腿保持
0.5–2.5 s  下蹲
2.5–4.5 s  恢复
4.5–5.5 s  最终保持
```

phase 边界不触发失败。唯一普通时间终止是 5.5 s episode timeout。

不奖励快，不惩罚慢。同样的最低点与最终高度，无论何时达到，总进度奖励相同。

## 初始状态

第一轮 100% 从已验证的单腿站立状态库开始：

```text
standing_prob=1.0
compressed_prob=0.0
airborne_prob=0.0
```

左右各 50%。

只有实际证据证明“会蹲但不会恢复”后，才允许新增少量 crouched reset。第一轮禁止
预先加入。

## 恢复高度带

每个 episode 使用自己的初始躯干高度 `z_start`，不使用跨 episode 固定高度。

完成高度条件：

```text
abs(z - z_start) <= 0.005 m
```

也就是初始高度周围 ±5 mm。

依据：

- 左站立状态库高度标准差约 1.2 mm；
- 右站立状态库高度标准差约 0.9 mm；
- ±5 mm 约为自然离散的 4 倍；
- 足够容纳正常波动；
- 又不会把明显半蹲算作恢复。

±5 mm 只用于 completion 判定，不是上升奖励封顶。恢复阶段超过初始高度仍继续获得
新的高度进度奖励。

## 唯一有效性 Gate

```text
valid =
  支撑脚接触地面
  AND 摆动脚不接触地面
  AND 身体其他部位不接触地面
```

不加入角度、速度、COM 或倾斜阈值。

如果支撑脚离地：

- 该步任务奖励为 0；
- 不额外惩罚；
- 重新接触后可以继续。

如果摆动脚或身体接触地面：

- 该步任务奖励为 0；
- 身体非脚接触作为失败终止；
- 不设置额外巨额失败罚分。

## Reward

### 下蹲新深度

下蹲 phase 记录有效状态中的最低高度：

```text
depth = max(z_start - z, 0)
reward_down = max(depth - best_depth_so_far, 0) / 0.01
```

含义：每创造新的 10 mm 下蹲深度，获得 1 个 reward unit。

不封顶。保持低位不重复给分。

### 恢复新高度

恢复 phase 从本局最低高度开始记录新的最高恢复高度：

```text
rise = max(z - z_lowest, 0)
reward_up = max(rise - best_rise_so_far, 0) / 0.01
```

含义：每创造新的 10 mm 恢复高度，获得 1 个 reward unit。

不封顶。超过初始高度仍继续计分。

### 最终恢复完成

最终 phase 中同时满足：

```text
valid
AND 本局有效下蹲深度至少 5 mm
AND abs(z - z_start) <= 5 mm
```

连续保持 0.5 s 后支付一次 completion event。

5 mm 下蹲门槛只用于防止“原地不动直到最终阶段”骗取完成奖励，复用同一个
±5 mm 高度容差，不增加关节角度或姿态目标。下蹲深度奖励仍然连续、不封顶；
正式晋级仍要求左右平均下蹲深度至少 10 mm。

### 第一轮权重

| Reward | Weight |
|---|---:|
| `crouch_depth_progress` | 5.0 |
| `return_height_progress` | 5.0 |
| `return_completion` | 10.0 |

第一轮无其他正 task reward。

## Termination

保留：

- episode timeout；
- terrain bounds；
- NaN；
- 45° fall termination；
- 身体非脚接触终止；
- 成功完成终止。

移除：

- root-height 下限；
- command phase 边界失败；
- 未达到某个下蹲深度失败；
- 支撑脚离地失败；
- 摆动脚短暂接触的额外巨额罚分。

## 模型与 PPO

第一轮：

```text
actor: warm-start strict-v4 model3750
critic: fresh
optimizer: fresh
iteration: 0
obs normalizer: inherit actor normalizer
new command columns: reset to neutral
action std: fixed 0.02
entropy: 0
learning rate: 5e-5
learning epochs: 1
schedule: fixed
symmetry mirror loss: enabled
```

原因：

- std 0.005 对新动态动作探索过小；
- 默认 std 1.0 或可学习 std 曾摧毁继承 actor；
- 0.02 是受控探索，不允许训练中膨胀；
- 站立能力通过每局初始/最终阶段自然重放；
- 第一轮不使用 teacher action anchor，避免旧静态 actor 阻止下蹲。

## 实现文件

已新增：

- `src/mjlab_microduck/tasks/microduck_single_leg_crouch_env_cfg.py`
- `tests/test_single_leg_crouch_cfg.py`
- `scripts/single_leg_crouch/eval.py`
- `scripts/single_leg_crouch/play_checkpoint.py`
- `scripts/single_leg_crouch/auto_eval_checkpoints.py`
- `scripts/single_leg_crouch/publish_chinese_dashboard.py`

已修改：

- `src/mjlab_microduck/tasks/mdp.py`
- `src/mjlab_microduck/tasks/__init__.py`
- `scripts/single_leg_jump/auto_view_latest_checkpoint.py`

禁止复用或继续修改旧 jump reward/state machine 来伪装成新任务。新任务必须拥有独立
experiment name、metrics 和 TensorBoard。

实现采用独立 `_slc_*` 状态，不修改旧 `_slj_*` 跳跃状态机。只复用已验证的 standing
状态库加载入口、左右支撑接触工具和内部 phase 时钟。

## Smoke Gate

恢复付费 GPU 前必须在本地完成：

- [x] cfg/test 全通过；
- [x] 64 env × 5 iteration smoke；
- [x] 61D actor obs；
- [x] 无 NaN；
- [x] checkpoint 可保存；
- [x] ONNX 可导出；
- [x] viewer 可固定左/右侧；
- [x] progress reward 保持时为 0；
- [x] 离地时 task reward 为 0 且无额外惩罚；
- [x] ±5 mm completion band 单元测试通过。

最终 smoke：

```text
run:
logs/rsl_rl/single_leg_crouch/2026-09-04_00-31-55_single_leg_crouch

model_4.pt SHA-256:
edc6f305ab3ee9b456bcd0948d8b03199e3b2e201fc27461eb4497c91c27a0e4

ONNX SHA-256:
1a18932eb86599acd5707814964de4f4952a3f33857a337351113b0874690294

tests:
244 passed, 1 skipped

smoke:
5/5 iterations
actor obs 61D
action 14D
action std 0.02
nan_state 0
```

smoke 只证明训练、保存、ONNX 和指标链路可运行，不作为行为晋级证据。

## 在线拒绝 Gate

第一批 checkpoint：

```text
model_25
model_50
model_75
model_99
```

runner 使用 0 基 iteration 编号，因此 `model_99` 是完成 100 个 learning iterations
后的首轮终点。

立即拒绝条件：

- 任一侧 depth 连续两个 gate 不增长；
- 任一侧 return completion 到 model50 仍为 0；
- 左右一侧提升、另一侧持续下降；
- 非脚接触率持续上升；
- 模型只下降不恢复；
- 模型只维持站立不下降；
- std、NaN 或 penalty 符号异常。

不等待单独固定评估才能拒绝。

## 固定晋级 Gate

每侧 128 deterministic episodes，100% standing start。

model99（100 轮终点）初始晋级门槛：

- 左右 completion 各至少 50/128；
- 左右平均最大下蹲深度各至少 10 mm；
- completed episodes 最终高度误差在 ±5 mm；
- 非脚接触率不高于 5%；
- 人眼确认不是倒地、摆动脚撑地或明显 viewer exploit。

这是最低晋级线，不是最终质量目标。

## 设备恢复

原实例：

```text
instance: 49723497
machine: 13428
GPU: RTX 4090
原总价: $0.3677777778/hour
当前 rentable: false
```

流程：

1. 代码、测试、smoke、父 checkpoint、rollback、监控命令全部 ready；
2. 执行 `vastai start instance 49723497`；
3. 30 s 内确认 `actual_status=running`；
4. 若停在 scheduling 超过 30 s，保持实例 stopped，不产生 GPU 费；
5. 启动同价/更低价 4090 watcher；
6. 新机器必须重新完成 SSH、CUDA、Warp 和 64×5 smoke；
7. 正式训练确认首个递增 iteration 后才标记 ACTIVE。

## 持续监督

必须同时运行：

- Vast 价格/实例监控；
- Trainer 进度监控；
- TensorBoard event 同步；
- checkpoint 原子同步；
- model25/50/75/99 本地固定评估；
- 最新 checkpoint 8080 自动展示。

每次汇报包含：

- iteration/target；
- steps/s；
- 左右 depth 趋势；
- 左右 return completion 趋势；
- 上一 gate、当前 gate 和 delta；
- improving / flat / regressing；
- 当前 8080 checkpoint、SHA 和状态；
- 当前 TensorBoard 地址。

## Rollback

训练代码 rollback：

```text
git commit: 在本执行文档与实现提交完成后填写
```

actor rollback：

```text
strict-v4 model3750
SHA-256: 4f445e8eaa482b64596d69044730aafc42d3ed0bd78f52ce92e0cfa95d67a526
```

设备 rollback：

```text
instance 49723497 保持 stopped，直到全部替代条件 ready。
```

## 状态

```text
DOC_READY
CODE_READY
DEVICE_STOPPED
TRAINING_NOT_STARTED
```
