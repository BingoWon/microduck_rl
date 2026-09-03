# 2026-09-03 Vast 付费 GPU 部署、训练连续性与事故复盘

## 目的

本文记录 Microduck 单腿跳训练在 Vast 上的真实部署过程、浪费来源、已核销问题和永久操作流程。实时状态仍以 Vast API、受管 PID、训练日志和 checkpoint 为准；本文不是第二份实时状态总账。

## 本次机器合同

```text
Vast instance: 49723497
GPU: NVIDIA GeForce RTX 4090
VRAM: 23028 MiB
价格: $0.3677777778/hour，含 100 GB 容器磁盘
镜像: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
项目 Python: 3.12.14
Torch: 2.9.1+cu128
Warp: 1.12.0
训练甜点: 8192 environments
```

实测完整训练吞吐：

| environments | 稳定中位 steps/s | 峰值显存 | 结论 |
|---:|---:|---:|---|
| 4096 | 49,943 | 6,019 MiB | 可快速恢复，但不是吞吐赢家 |
| 8192 | 69,336，后续约 71k–76k | 12,663 MiB | 当前可用甜点 |
| 12288 | 超过 7 分钟未进入 iteration | 初始化约 4,555 MiB | 淘汰 |

甜点按有效 steps/s 和 time-to-quality 裁决，不以显存占满为目标。

## 已发生事故

### 宿主镜像路径故障

实例 `49722171` 在新旧两个 PyTorch 镜像上都长期停在同一个 Docker checksum 状态，容器不存在且 SSH 不可用。说明问题属于宿主 Docker/存储路径，而不是镜像版本。该实例已销毁，机器 `38442` 必须保持黑名单。

### 把初始化误称为训练

8192 环境构建会长时间占用单个 CPU 核，GPU 利用率为 0。只有出现持续递增的 `Learning iteration` 且 GPU 执行后才能称为训练。PID 存活、显存分配或 Vast `actual_status=running` 都不够。

### 手工 gate 导致零 Trainer 窗口

多次使用有限 `max_iterations` gate，结束后才人工决定续跑，造成重复 8192 初始化和 GPU 空闲。以后 gate 必须在结束前准备好可晋级 continuation；checkpoint 出现后自动继续，评估在本地异步执行。

### 评估错误地暂停远端训练

checkpoint 约 4.6 MB，本地 CPU evaluator 已经被证明可用。常规评估应同步 checkpoint 后本地异步运行，不占用远端 GPU，也不暂停 Trainer。只有最终验收需要远端 GPU 交叉复核。

### 固定命令侧但未固定 reset 侧

旧 `eval_single_leg_jump.py` 和 recorder 只设置 command `fixed_side`，却没有同步设置 `reset_single_leg_jump.fixed_side`。standing bank reset 会覆盖 command side，导致左右评估标签失真。现已在 canonical evaluator 和 recorder 中同时固定两者。

### 默认 PPO 摧毁 v6

accepted v6 的动作标准差为 `0.005`。默认 jump 配置继承 stand 的 `learning_rate=1e-3`、`entropy_coef=0.01` 和可学习 std；model250 的 std 膨胀到约 `0.025–0.031`，固定评估左右均为 0 起跳。现已锁定 std 为 `0.005`，关闭 entropy，基础学习率降为 `5e-5`；实际保守 gate 使用 `1e-6` 和一个 learning epoch。

### 训练指标不能代替确定性验收

训练日志出现约 3%–5% completion 时，确定性 evaluator 仍可能为 0。训练指标包含随机动作采样和 reverse-reset 分布，只能用于诊断。晋级必须依据固定 standing start、固定 side、确定性 actor 的真实起跳、同脚落地和恢复。

### SSH 与监控故障

- `ssh2.vast.ai` 曾发生本地 DNS 解析失败；通过 `1.1.1.1` 得到代理 IP，并保留 `vastai ssh-url` 给出的直连 IP/端口。
- 训练监控曾遗漏 `root@`，使用本地用户名连接而失败。
- 训练监控曾在一次 SSH timeout 后退出；现已改为记录 attention 并继续重试。
- macOS Documents 路径读取 checkpoint 曾触发 `torch.load()` 超时；评估 checkpoint 先原子下载到 `/tmp`。

## 当前能力证据

修正后的固定侧 evaluator，每侧 128 episodes：

| checkpoint | 左：起跳/落地/完成 | 右：起跳/落地/完成 |
|---|---:|---:|
| accepted v6 | 29/28/4 | 7/3/2 |
| 默认 PPO model250 | 0/0/0 | 0/0/0 |
| 默认 PPO model500 | 0/0/0 | 3/3/0 |
| locked std model250 | 0/0/0 | 0/0/0 |
| ultra-conservative model250 | 39/38/3 | 23/18/2 |
| ultra-conservative model500 | 38/37/0 | 23/14/0 |

ultra model250 证明极低学习率可以保留并增强起跳/落地，但 model500 的恢复完成归零，说明纯 PPO 对恢复能力仍不稳定。当前 height-off continuation 只用于验证“移除过早高度优化是否恢复 completion”，不能无限续跑；下一 checkpoint 仍为 0 completion 时，纯 PPO 路线失效，必须切换到行为锚定/恢复示范候选。

## 固定部署顺序

1. 从干净 commit 生成 `git archive`，不传 Git 历史、logs 或整个 1.4 GB 工作目录。
2. 单独传 checkpoint、manifest 和 SHA-256；远端逐文件校验。
3. 使用已验证镜像，安装 Python 3.12，执行 `uv sync --frozen`。
4. 验证 `torch.cuda.is_available()`、GPU 数量、Warp CUDA device、任务注册。
5. 64 environments × 5 iterations smoke 只在新代码/新环境合同首次部署时执行。
6. 正式 worlds 选择使用完整训练 iteration 的 steps/s，不使用孤立微基准。
7. 启动一个价格监控和一个训练监控；监控状态与训练状态分离。
8. 当前 Trainer 首次有效 iteration 后，立即准备下一候选及回滚 checkpoint。
9. checkpoint 本地原子同步：写 `.part`，校验完整后 rename。
   每个 remote run 必须使用独立子目录，禁止不同 run 的
   `model_250.pt`、`model_500.pt` 相互覆盖。
10. evaluator 从 `/tmp` checkpoint 本地异步运行，不暂停远端训练。
11. 候选在 checkpoint 边界完成原子切换；失败立即恢复旧 checkpoint。
12. 实例销毁前同步全部关键资产并验证归档 SHA-256。

## 永久汇报格式

每次训练汇报第一行必须包含：

```text
iteration/target | steps/s | GPU util | VRAM | nan_state
```

随后必须报告固定 evaluator 的：

```text
left: takeoff / same-foot landing / completion
right: takeoff / same-foot landing / completion
```

没有新确定性评估时，明确写“尚无新固定评估”，不得用训练 reward 或随机 completion 替代。
