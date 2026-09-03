#!/usr/bin/env python3
"""Publish a compact Chinese TensorBoard dashboard from raw training events."""

import argparse
import json
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

METRICS = {
    "Evaluation/left/success_rate": ("01_固定评估_最终结果/左腿_完整成功率", 1.0),
    "Evaluation/right/success_rate": ("01_固定评估_最终结果/右腿_完整成功率", 1.0),
    "Evaluation/left/recovery_completion_rate": (
        "01_固定评估_最终结果/左腿_落地后恢复率",
        1.0,
    ),
    "Evaluation/right/recovery_completion_rate": (
        "01_固定评估_最终结果/右腿_落地后恢复率",
        1.0,
    ),
    "Evaluation/left/true_takeoff_rate": ("02_固定评估_动作链/左腿_真实起跳率", 1.0),
    "Evaluation/right/true_takeoff_rate": ("02_固定评估_动作链/右腿_真实起跳率", 1.0),
    "Evaluation/left/same_foot_landing_rate": (
        "02_固定评估_动作链/左腿_同脚落地率",
        1.0,
    ),
    "Evaluation/right/same_foot_landing_rate": (
        "02_固定评估_动作链/右腿_同脚落地率",
        1.0,
    ),
    "Evaluation/left/swing_foot_contact_rate": (
        "02_固定评估_动作链/左腿_错误脚接触率",
        1.0,
    ),
    "Evaluation/right/swing_foot_contact_rate": (
        "02_固定评估_动作链/右腿_错误脚接触率",
        1.0,
    ),
    "Evaluation/left/peak_root_height_gain_m/mean": (
        "03_固定评估_跳跃高度/左腿_平均高度毫米",
        1000.0,
    ),
    "Evaluation/right/peak_root_height_gain_m/mean": (
        "03_固定评估_跳跃高度/右腿_平均高度毫米",
        1000.0,
    ),
    "Evaluation/left/peak_root_height_gain_m/p50": (
        "03_固定评估_跳跃高度/左腿_中位高度毫米",
        1000.0,
    ),
    "Evaluation/right/peak_root_height_gain_m/p50": (
        "03_固定评估_跳跃高度/右腿_中位高度毫米",
        1000.0,
    ),
    "Evaluation/left/peak_root_height_gain_m/p100": (
        "03_固定评估_跳跃高度/左腿_最高高度毫米",
        1000.0,
    ),
    "Evaluation/right/peak_root_height_gain_m/p100": (
        "03_固定评估_跳跃高度/右腿_最高高度毫米",
        1000.0,
    ),
    "Episode_Metrics/jump_takeoff_rate_left": ("04_训练趋势_动作链/左腿_起跳率", 1.0),
    "Episode_Metrics/jump_takeoff_rate_right": ("04_训练趋势_动作链/右腿_起跳率", 1.0),
    "Episode_Metrics/jump_landing_rate_left": ("04_训练趋势_动作链/左腿_落地率", 1.0),
    "Episode_Metrics/jump_landing_rate_right": ("04_训练趋势_动作链/右腿_落地率", 1.0),
    "Episode_Metrics/jump_completion_rate_left": ("04_训练趋势_动作链/左腿_完成率", 1.0),
    "Episode_Metrics/jump_completion_rate_right": ("04_训练趋势_动作链/右腿_完成率", 1.0),
    "Episode_Metrics/jump_failure_rate_left": ("04_训练趋势_动作链/左腿_失败率", 1.0),
    "Episode_Metrics/jump_failure_rate_right": ("04_训练趋势_动作链/右腿_失败率", 1.0),
    "Episode_Metrics/jump_peak_height_gain_left": (
        "05_训练趋势_高度与站立/左腿_平均高度毫米",
        1000.0,
    ),
    "Episode_Metrics/jump_peak_height_gain_right": (
        "05_训练趋势_高度与站立/右腿_平均高度毫米",
        1000.0,
    ),
    "Episode_Metrics/single_leg_success_left": (
        "05_训练趋势_高度与站立/左腿_单腿站成功率",
        1.0,
    ),
    "Episode_Metrics/single_leg_success_right": (
        "05_训练趋势_高度与站立/右腿_单腿站成功率",
        1.0,
    ),
    "Episode_Reward/jump_completion": ("06_奖励构成/完整成功奖励", 1.0),
    "Episode_Reward/jump_height": ("06_奖励构成/跳跃高度奖励", 1.0),
    "Episode_Reward/jump_recovery_progress": ("06_奖励构成/落地恢复进度奖励", 1.0),
    "Episode_Reward/jump_takeoff": ("06_奖励构成/真实起跳奖励", 1.0),
    "Episode_Reward/jump_landing": ("06_奖励构成/同脚落地奖励", 1.0),
    "Episode_Reward/swing_contact": ("06_奖励构成/错误脚接触惩罚", 1.0),
    "Episode_Termination/nan_state": ("07_安全与效率/异常数值终止", 1.0),
    "Perf/total_fps": ("07_安全与效率/每秒环境步数", 1.0),
    "Policy/mean_std": ("07_安全与效率/动作探索标准差", 1.0),
    "Train/mean_reward": ("07_安全与效率/平均总奖励", 1.0),
    "Train/mean_episode_length": ("07_安全与效率/平均回合长度", 1.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "发布状态.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    writer = SummaryWriter(args.output)

    while True:
        accumulator = EventAccumulator(str(args.source))
        accumulator.Reload()
        available = set(accumulator.Tags()["scalars"])
        changed = False
        for source_tag, (target_tag, scale) in METRICS.items():
            if source_tag not in available:
                continue
            last_step = int(state.get(source_tag, -1))
            for event in accumulator.Scalars(source_tag):
                if event.step <= last_step:
                    continue
                writer.add_scalar(target_tag, event.value * scale, event.step)
                state[source_tag] = event.step
                changed = True
        if changed:
            writer.flush()
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(state_path)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
