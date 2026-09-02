#!/usr/bin/env python3
"""Fit BAM parameters using BOTH position and present_current residuals.

bam/fit.py scores on position only (`compute_score`, fit.py:61) and discards
`present_current` -- the one channel that measures torque directly. A Fisher
analysis of this bench put adding it at 3.4x on the downstream torque number at
a realistic sigma_I of 100 mA, which beats every trajectory-design decision
combined and needs no hardware.

Simulated current is recovered exactly from what `rollout_log` already returns:
`step()` uses `compute_torque(control, enable, q + q_offset, dq)`, and for a DC
motor tau = kt*V/R - kt^2*dq/R, so I_sim = tau / kt. positions/velocities/controls
are the same values `step` consumed, so this reconstructs it without touching
BAM internals.

  uv run python scripts/bam_fit_current.py --logdir logs/ \
      --actuator xl330 --model m6 --trials 20000 --workers 8 \
      --output params_new.json --validation-kp 800 \
      --set "{'load_friction_motor_quad':0.0,'load_friction_external_quad':0.0,
              'load_friction_external':0.0,'load_friction_motor_stribeck':0.0}"
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy

import numpy as np
import optuna

from bam import simulate
from bam.actuators import actuators
from bam.logs import Logs
from bam.model import Model, models, load_model

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--logdir", required=True)
ap.add_argument("--actuator", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--output", default="params_current.json")
ap.add_argument("--trials", type=int, default=20000)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--reset-period", type=float, default=None)
ap.add_argument("--set", type=str, default="")
ap.add_argument("--validation-kp", type=int, default=0)
ap.add_argument("--current-weight", type=float, default=1.0,
                help="lambda in score = pos_mae[rad] + lambda * cur_mae[A]. "
                     "Both terms land in ~0.01-0.1, so 1.0 balances them; "
                     "0 reproduces bam/fit.py's position-only objective.")
ap.add_argument("--restarts", type=int, default=4,
                help="independent studies, best kept. Monte-Carlo found 1 in "
                     "5-10 refits landing in a local minimum at 500x the correct "
                     "cost even from near-truth starts, so this is not optional.")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

logs = Logs(args.logdir)
validation = None
if args.validation_kp > 0:
    validation = logs.split(args.validation_kp)
    print(f"held out kp={args.validation_kp}: {len(validation.logs)} logs, "
          f"{len(logs.logs)} remain for fitting")
print(f"fitting on {len(logs.logs)} logs from {args.logdir}")

n_with_current = sum(
    1 for lg in logs.logs if any("current" in e for e in lg["entries"])
)
print(f"  {n_with_current}/{len(logs.logs)} logs carry a current channel")
if args.current_weight > 0 and n_with_current == 0:
    raise SystemExit("--current-weight > 0 but no log has a 'current' field; "
                     "re-record with a recorder that logs present_current.")


def make_model() -> Model:
    model = models[args.model]()
    model.set_actuator(actuators[args.actuator]())
    if args.set != "":
        for key, value in eval(args.set).items():
            model.get_parameters()[key].value = value
            model.get_parameters()[key].optimize = False
    return model


def score_log(model: Model, log: dict) -> tuple[float, float]:
    """Return (position MAE [rad], current MAE [A]); current MAE is nan if the
    log has no current channel."""
    sim = simulate.Simulator(model)
    positions, velocities, controls = sim.rollout_log(
        log, reset_period=args.reset_period, simulate_control=True
    )
    entries = log["entries"]
    pos_mae = float(np.mean(np.abs(
        np.array(positions) - np.array([e["position"] for e in entries])
    )))

    idx = [i for i, e in enumerate(entries) if "current" in e]
    if not idx or args.current_weight == 0.0:
        return pos_mae, float("nan")

    # I_sim = tau / kt, with tau exactly as step() computed it.
    kt = model.kt.value
    act = model.actuator
    q_off = model.q_offset.value
    sim_i, real_i = [], []
    for i in idx:
        tau = act.compute_torque(
            controls[i], entries[i]["torque_enable"], positions[i] + q_off,
            velocities[i],
        )
        sim_i.append(float(tau) / kt)
        real_i.append(entries[i]["current"])
    cur_mae = float(np.mean(np.abs(np.array(sim_i) - np.array(real_i))))
    return pos_mae, cur_mae


def total_score(model: Model, log_set: Logs) -> tuple[float, float, float]:
    p = c = 0.0
    n_c = 0
    for log in log_set.logs:
        pm, cm = score_log(model, log)
        p += pm
        if cm == cm:  # not nan
            c += cm
            n_c += 1
    p /= len(log_set.logs)
    c = c / n_c if n_c else 0.0
    return p + args.current_weight * c, p, c


def objective(trial):
    model = make_model()
    for name, param in model.get_parameters().items():
        if param.optimize:
            param.value = trial.suggest_float(name, param.min, param.max)
    return total_score(model, logs)[0]


best = None
for r in range(args.restarts):
    sampler = optuna.samplers.CmaEsSampler(seed=args.seed + r)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials // args.restarts,
                   n_jobs=args.workers, show_progress_bar=True)
    print(f"  restart {r}: best {study.best_value:.6f}")
    if best is None or study.best_value < best.best_value:
        best = study

model = make_model()
data = deepcopy(best.best_params)
for key, param in model.get_parameters().items():
    if key not in data:
        data[key] = param.value
    else:
        param.value = data[key]
data["model"] = args.model
data["actuator"] = args.actuator
json.dump(data, open(args.output, "w"), indent=2)

tot, p, c = total_score(model, logs)
print(f"\nfit    : score {tot:.6f}  (position {p*1000:.3f} mrad, "
      f"current {c*1000:.2f} mA)")
if validation is not None and validation.logs:
    vt, vp, vc = total_score(model, validation)
    print(f"held-out kp={args.validation_kp}: score {vt:.6f}  "
          f"(position {vp*1000:.3f} mrad, current {vc*1000:.2f} mA)")
    print("  -> if held-out is close to fit, the model generalises across gain;"
          "\n     if it is much worse, extrapolating to other kp is not justified.")
print(f"wrote {args.output}")
