#!/usr/bin/env python3
"""Publish a compact Chinese TensorBoard dashboard from raw training events."""

import argparse
import json
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

METRICS = {
    "Evaluation/left/success_rate": ("01_核心表现/固定评估_左腿完整成功率", 1.0),
    "Evaluation/right/success_rate": ("01_核心表现/固定评估_右腿完整成功率", 1.0),
    "Evaluation/left/recovery_completion_rate": (
        "03_恢复与站立/固定评估_左腿落地恢复率",
        1.0,
    ),
    "Evaluation/right/recovery_completion_rate": (
        "03_恢复与站立/固定评估_右腿落地恢复率",
        1.0,
    ),
    "Evaluation/left/true_takeoff_rate": ("02_动作链/固定评估_左腿起跳率", 1.0),
    "Evaluation/right/true_takeoff_rate": ("02_动作链/固定评估_右腿起跳率", 1.0),
    "Evaluation/left/same_foot_landing_rate": (
        "02_动作链/固定评估_左腿同脚落地率",
        1.0,
    ),
    "Evaluation/right/same_foot_landing_rate": (
        "02_动作链/固定评估_右腿同脚落地率",
        1.0,
    ),
    "Evaluation/left/swing_foot_contact_rate": (
        "02_动作链/固定评估_左腿错误脚接触率",
        1.0,
    ),
    "Evaluation/right/swing_foot_contact_rate": (
        "02_动作链/固定评估_右腿错误脚接触率",
        1.0,
    ),
    "Evaluation/left/peak_root_height_gain_m/mean": (
        "01_核心表现/固定评估_左腿平均跳高毫米",
        1000.0,
    ),
    "Evaluation/right/peak_root_height_gain_m/mean": (
        "01_核心表现/固定评估_右腿平均跳高毫米",
        1000.0,
    ),
    "Evaluation/left/peak_root_height_gain_m/p50": (
        "01_核心表现/固定评估_左腿中位跳高毫米",
        1000.0,
    ),
    "Evaluation/right/peak_root_height_gain_m/p50": (
        "01_核心表现/固定评估_右腿中位跳高毫米",
        1000.0,
    ),
    "Evaluation/left/peak_root_height_gain_m/p100": (
        "01_核心表现/固定评估_左腿最高跳高毫米",
        1000.0,
    ),
    "Evaluation/right/peak_root_height_gain_m/p100": (
        "01_核心表现/固定评估_右腿最高跳高毫米",
        1000.0,
    ),
    "Episode_Metrics/jump_takeoff_rate_left": ("02_动作链/训练采样_左腿起跳率", 1.0),
    "Episode_Metrics/jump_takeoff_rate_right": ("02_动作链/训练采样_右腿起跳率", 1.0),
    "Episode_Metrics/jump_landing_rate_left": ("02_动作链/训练采样_左腿落地率", 1.0),
    "Episode_Metrics/jump_landing_rate_right": ("02_动作链/训练采样_右腿落地率", 1.0),
    "Episode_Metrics/jump_completion_rate_left": ("01_核心表现/训练采样_左腿完整成功率", 1.0),
    "Episode_Metrics/jump_completion_rate_right": ("01_核心表现/训练采样_右腿完整成功率", 1.0),
    "Episode_Metrics/jump_failure_rate_left": ("02_动作链/训练采样_左腿失败率", 1.0),
    "Episode_Metrics/jump_failure_rate_right": ("02_动作链/训练采样_右腿失败率", 1.0),
    "Episode_Metrics/jump_peak_height_gain_left": (
        "01_核心表现/训练采样_左腿平均跳高毫米",
        1000.0,
    ),
    "Episode_Metrics/jump_peak_height_gain_right": (
        "01_核心表现/训练采样_右腿平均跳高毫米",
        1000.0,
    ),
    "Episode_Metrics/single_leg_success_left": (
        "03_恢复与站立/训练采样_左腿单腿站成功率",
        1.0,
    ),
    "Episode_Metrics/single_leg_success_right": (
        "03_恢复与站立/训练采样_右腿单腿站成功率",
        1.0,
    ),
    "Episode_Reward/jump_completion": ("04_奖励构成/完整成功奖励", 1.0),
    "Episode_Reward/jump_height": ("04_奖励构成/跳跃高度奖励", 1.0),
    "Episode_Reward/jump_recovery_progress": ("04_奖励构成/落地恢复进度奖励", 1.0),
    "Episode_Reward/jump_takeoff": ("04_奖励构成/真实起跳奖励", 1.0),
    "Episode_Reward/jump_landing": ("04_奖励构成/同脚落地奖励", 1.0),
    "Episode_Reward/swing_contact": ("04_奖励构成/错误脚接触惩罚", 1.0),
    "Episode_Termination/nan_state": ("05_训练稳定性/异常数值终止", 1.0),
    "Perf/total_fps": ("05_训练稳定性/每秒环境步数", 1.0),
    "Policy/mean_std": ("05_训练稳定性/动作探索标准差", 1.0),
    "Loss/teacher_anchor": ("05_训练稳定性/v6教师动作偏离均方误差", 1.0),
    "Train/mean_reward": ("05_训练稳定性/平均总奖励", 1.0),
    "Train/mean_episode_length": ("05_训练稳定性/平均回合长度", 1.0),
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
