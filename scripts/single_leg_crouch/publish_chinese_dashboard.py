#!/usr/bin/env python3
"""Publish a compact Chinese TensorBoard view for crouch decisions."""

import argparse
import json
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

METRICS = {
    "Evaluation/left/success_rate": ("01_完成率/固定评估_左腿", 1.0),
    "Evaluation/right/success_rate": ("01_完成率/固定评估_右腿", 1.0),
    "Episode_Metrics/crouch_completion_rate_left": (
        "01_完成率/训练采样_左腿",
        1.0,
    ),
    "Episode_Metrics/crouch_completion_rate_right": (
        "01_完成率/训练采样_右腿",
        1.0,
    ),
    "Evaluation/left/max_depth_m/mean": (
        "02_下蹲深度/固定评估_左腿平均毫米",
        1000.0,
    ),
    "Evaluation/right/max_depth_m/mean": (
        "02_下蹲深度/固定评估_右腿平均毫米",
        1000.0,
    ),
    "Episode_Metrics/crouch_max_depth_left": (
        "02_下蹲深度/训练采样_左腿毫米",
        1000.0,
    ),
    "Episode_Metrics/crouch_max_depth_right": (
        "02_下蹲深度/训练采样_右腿毫米",
        1000.0,
    ),
    "Evaluation/left/max_rise_m/mean": (
        "03_恢复高度/固定评估_左腿平均毫米",
        1000.0,
    ),
    "Evaluation/right/max_rise_m/mean": (
        "03_恢复高度/固定评估_右腿平均毫米",
        1000.0,
    ),
    "Evaluation/left/final_height_error_m/mean": (
        "03_恢复高度/固定评估_左腿最终误差毫米",
        1000.0,
    ),
    "Evaluation/right/final_height_error_m/mean": (
        "03_恢复高度/固定评估_右腿最终误差毫米",
        1000.0,
    ),
    "Evaluation/left/nonfoot_contact_rate": (
        "04_失败/固定评估_左腿身体触地率",
        1.0,
    ),
    "Evaluation/right/nonfoot_contact_rate": (
        "04_失败/固定评估_右腿身体触地率",
        1.0,
    ),
    "Evaluation/left/fell_over_rate": (
        "04_失败/固定评估_左腿倾倒率",
        1.0,
    ),
    "Evaluation/right/fell_over_rate": (
        "04_失败/固定评估_右腿倾倒率",
        1.0,
    ),
    "Episode_Metrics/crouch_failure_rate_left": (
        "04_失败/训练采样_左腿身体触地率",
        1.0,
    ),
    "Episode_Metrics/crouch_failure_rate_right": (
        "04_失败/训练采样_右腿身体触地率",
        1.0,
    ),
    "Episode_Reward/crouch_depth_progress": (
        "05_奖励/下蹲深度进度",
        1.0,
    ),
    "Episode_Reward/return_height_progress": (
        "05_奖励/恢复高度进度",
        1.0,
    ),
    "Episode_Reward/return_completion": ("05_奖励/完整恢复", 1.0),
    "Episode_Termination/nan_state": ("06_训练健康/异常数值终止", 1.0),
    "Policy/mean_std": ("06_训练健康/动作探索标准差", 1.0),
    "Perf/total_fps": ("06_训练健康/每秒环境步数", 1.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
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
        if args.once:
            writer.close()
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
