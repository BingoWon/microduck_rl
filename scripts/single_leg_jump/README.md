# Single-leg jump operations

Canonical reusable tools for the single-leg policy family:

- `play_checkpoint.py`: interactive four-command Viser viewer.
- `eval_jump.py`: fixed-side deterministic jump evaluation.
- `eval_transitions.py`: six-transaction evaluation.
- `harvest_reset_states.py`: validated reset-state harvesting.
- `record_jump.py`: fixed-side rollout recording.
- `eval_stand.py` and `record_stand.py`: stand-policy evaluation and recording.
- `log_eval_tensorboard.py`: append evaluation results to TensorBoard.
- `training_status.py`: one JSON snapshot of remote iteration, throughput,
  GPU, VRAM, power, and NaN state.
- `sync_remote_tensorboard.py`: continuously mirror only the active remote
  TensorBoard event file while checkpoints use their separate atomic sync.
- `auto_eval_checkpoints.py`: evaluate synchronized model250/500/750/1000
  checkpoints locally and publish deployment-style metrics into TensorBoard.
- `publish_chinese_dashboard.py`: rewrite the raw event stream into a compact,
  Chinese-only decision dashboard.

Cloud operations and dated evidence remain outside the repository under:

```text
/Users/bingo/Documents/Codex/2026-09-02/wo-x/work/cloud/
```

The reusable cloud scripts there are:

- `wait_for_microduck_vast.py`
- `monitor_vast.py`
- `monitor_training.py`

The command entries above are symlinks to the existing canonical scripts.
Keeping one implementation prevents later fixes from diverging across copies.
