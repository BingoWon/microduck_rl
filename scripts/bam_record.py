#!/usr/bin/env python3
"""Record a BAM identification trajectory from the real XL330 bench.

Writes the raw log format bam/fit.py consumes (via bam/process.py). One file
per (trajectory, payload, kp, vin) configuration.

  # single run, no payload, arm only
  uv run python scripts/bam_record.py --trajectory sin_time_square \
      --arm-mass 0.031 --length 0.160 --mass 0.0 --kp 400 --out raw/

  # then resample and fit
  uv run python -m bam.process --raw raw/ --logdir logs/ --dt 0.005
  uv run python -m bam.fit --logdir logs/ --actuator xl330 --model m6 \
      --trials 20000 --output params_new.json

Safety: goals are clamped to --limit-deg, the run aborts if the arm is measured
beyond it, and torque is always released on exit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from rustypot import Xl330PyController

DXL_VEL_TICK_TO_RAD_S = 0.229 * 2.0 * math.pi / 60.0
LOG_HZ = 200.0

# XL330-M288 RATED stall torque at its rated 5 V. This is the GEARBOX's design
# figure and is the number that matters mechanically -- not the torque the motor
# can actually produce, which at 8.4 V is ~1.6x higher, and not the "usable"
# torque left after friction. A 272 g payload at 160 mm (0.451 N.m = 87% of
# stall) STRIPPED A GEARBOX. Dynamic overshoot on the fast trajectories adds
# shock well above the static figure, so the cap is deliberately conservative.
XL330_STALL_NM = 0.52
KT_NM_PER_A = 0.366  # bam params/xl330/m6.json, joint-referred

# --- trajectories -----------------------------------------------------------
# BAM's five, plus the two the Fisher-information study added for a +-90 deg
# bench. up_and_down / lift_and_drop are scaled to +-85 deg: unscaled they sit
# exactly on the mechanical limit.
_A85 = math.radians(85.0) / (math.pi / 2)  # 0.944


def _cubic(keyframes, t):
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]
    for i in range(len(keyframes) - 1):
        t0, x0, v0 = keyframes[i]
        t1, x1, v1 = keyframes[i + 1]
        if t0 <= t <= t1:
            d = t1 - t0
            s = (t - t0) / d
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            return h00 * x0 + h10 * d * v0 + h01 * x1 + h11 * d * v1
    return keyframes[-1][1]


def sin_time_square(t):
    return math.sin(t**2), True


def up_and_down(t):
    kf = [[0.0, 0.0, 0.0],
          [3.0, _A85 * math.pi / 2, 0.0],
          [6.0, _A85 * 0.8 * math.pi / 2, 0.0]]
    return _cubic(kf, t), True


def lift_and_drop(t):
    kf = [[0.0, 0.0, 0.0], [2.0, -_A85 * math.pi / 2, 0.0]]
    return _cubic(kf, t), t < 2.0


def nothing(t):
    return 0.0, False


def sin_sin_scaled(t):
    """BAM's sin_sin with the CARRIER scaled to 0.82.

    Unscaled it reaches +-100.8 deg. The amplitude is nearly all carrier, while
    the fast modulated term (+-28.6 deg) carries 2.43 of the 3.67 rad/s peak
    speed -- so scaling the carrier costs ~5% of the velocity content and
    nothing measurable on load_friction_motor. s=0.82 commands +-87.4 deg.
    """
    s = 0.82
    return s * math.sin(t) * math.pi / 2 + math.sin(5.0 * t) * 0.5 * math.sin(2.0 * t), True


def chirp(t):
    """60 deg, 0.2 -> 5.0 Hz linear sweep over 6 s, 0.5 s fade-in.

    Amplitude buys nothing here (the Fisher study found 60-75 deg identical);
    upper sweep frequency does. Started at 70 deg, which MEASURED -73.5/+75.1 at
    kp=400 but -82.3/+86.2 at kp=800 -- only 3.8 deg from the 90 deg abort. Since
    amplitude is free, dropped to 60 deg. Do not raise it: a naive +-85 deg chirp
    overshoots to -96 deg.
    """
    f0, f1, dur = 0.2, 5.0, 6.0
    k = (f1 - f0) / dur
    phase = 2.0 * math.pi * (f0 * t + 0.5 * k * t * t)
    fade = min(1.0, t / 0.5)
    return math.radians(60.0) * fade * math.sin(phase), True


def make_staircase(max_deg: float, n: int = 5, ramp: float = 0.7, dwell: float = 1.0):
    """Bidirectional quasi-static staircase: up through n angles, then back down.

    Friction opposes motion, so at a given angle the holding torque differs
    depending on which side it was approached from. The MEAN of the two cancels
    friction and gives the gravity torque (hence the arm's first moment m*d);
    HALF THE DIFFERENCE is the friction at that load. That makes it the only
    excitation here that measures `load_friction_motor` directly rather than
    inferring it through a payload proxy.

    Quasi-static, so currents stay in the 0.02-0.15 A range -- the safest run in
    the set. Angle must be capped per payload (75 deg to 68 g, 55 deg at 89 g):
    the holding torque is what loads the gearbox and it grows as sin(theta).
    """
    angles = [max_deg * (i + 1) / n for i in range(n)]
    seq = [0.0] + angles + angles[-2::-1] + [0.0]   # up then down, no repeat at apex
    seg = ramp + dwell
    duration = len(seq) * seg

    def traj(t: float):
        k = min(int(t / seg), len(seq) - 1)
        frac = (t - k * seg) / ramp if ramp > 0 else 1.0
        prev = seq[k - 1] if k > 0 else 0.0
        if frac >= 1.0:
            ang = seq[k]                       # dwell: hold, this is the sample
        else:
            a = 0.5 * (1.0 - math.cos(math.pi * frac))
            ang = prev + (seq[k] - prev) * a   # cosine ramp between plateaus
        return math.radians(ang), True

    return traj, duration


TRAJECTORIES = {
    "sin_time_square": (sin_time_square, 6.0),
    "up_and_down": (up_and_down, 6.0),
    "lift_and_drop": (lift_and_drop, 6.0),
    "nothing": (nothing, 6.0),
    "sin_sin_scaled": (sin_sin_scaled, 6.0),
    "chirp": (chirp, 6.0),
}

# The constant-amplitude chirp draws 140% of RATED STALL TORQUE (2.00 A peak,
# 0.73 N.m) in its 3.4-5.0 Hz third, and sits above 1.2 A for 12% of the run.
# Torque demand scales with acceleration, which for a fixed amplitude grows as
# f^2 -- so a constant-amplitude sweep is inherently damaging at its top end.
# It very likely contributed to the first gearbox failure. Requires --i-know.
UNSAFE_TRAJECTORIES = {"chirp"}


def scalar(x) -> float:
    return float(x[0]) if isinstance(x, (list, tuple)) else float(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--id", type=int, default=15, dest="motor_id")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--trajectory", required=True,
                    choices=sorted(list(TRAJECTORIES) + ["staircase"]))
    ap.add_argument("--kp", type=int, required=True)
    ap.add_argument("--mass", type=float, required=True,
                    help="payload mass at the arm tip [kg] (WEIGH IT)")
    ap.add_argument("--arm-mass", type=float, required=True,
                    help="printed arm mass [kg] (WEIGH IT - bam defaults this to 0)")
    ap.add_argument("--length", type=float, required=True, help="arm length [m]")
    ap.add_argument("--vin", type=float, default=None,
                    help="supply volts; default = read from the servo")
    ap.add_argument("--cmd-limit-deg", type=float, default=88.0,
                    help="goals are clamped here (default 88)")
    ap.add_argument("--abort-deg", type=float, default=90.0,
                    help="abort if the ACHIEVED position exceeds this (default 90)")
    ap.add_argument("--out", default="raw")
    ap.add_argument("--stair-max-deg", type=float, default=75.0,
                    help="staircase apex angle. Cap per payload: 75 to 68 g, "
                         "55 at 89 g -- holding torque grows as sin(theta).")
    ap.add_argument("--i-know", action="store_true",
                    help="run a trajectory flagged unsafe (see UNSAFE_TRAJECTORIES)")
    ap.add_argument("--max-torque-frac", type=float, default=0.5,
                    help="refuse to run if peak gravity torque exceeds this "
                         "fraction of the RATED stall torque (default 0.5)")
    ap.add_argument("--max-current", type=float, default=1.0,
                    help="abort if current exceeds this for 2 consecutive checks [A]")
    args = ap.parse_args()

    if args.trajectory in UNSAFE_TRAJECTORIES and not args.i_know:
        print(f"REFUSING '{args.trajectory}': measured 2.00 A peak = 140% of rated "
              "stall torque, sustained >1.2 A for 12% of the run.\n"
              "  Torque demand grows as f^2 at fixed amplitude, so the top of the "
              "sweep is the problem.\n"
              "  Needs an amplitude-tapered sweep (a ~1/f^2 envelope) before it is "
              "safe. Pass --i-know to override.")
        return 9

    if args.trajectory == "staircase":
        traj, duration = make_staircase(args.stair_max_deg)
    else:
        traj, duration = TRAJECTORIES[args.trajectory]
    cmd_limit = math.radians(args.cmd_limit_deg)
    abort_at = math.radians(args.abort_deg)

    # --- mechanical guard: gearbox, not control authority -------------------
    tau_peak = (args.mass + args.arm_mass / 2.0) * 9.81 * args.length
    frac = tau_peak / XL330_STALL_NM
    cap = args.max_torque_frac * XL330_STALL_NM
    print(f"peak gravity torque {tau_peak:.3f} N.m = {frac*100:.0f}% of rated stall "
          f"({XL330_STALL_NM:.2f} N.m); cap {cap:.3f} N.m")
    if tau_peak > cap:
        m_ok = cap / (9.81 * args.length) - args.arm_mass / 2.0
        print(f"  REFUSING: {frac*100:.0f}% of rated stall exceeds the "
              f"{args.max_torque_frac*100:.0f}% cap.")
        print(f"  At length={args.length:.3f} m the maximum payload is "
              f"{m_ok*1000:.0f} g. Reduce the mass or shorten the arm.")
        print("  (272 g at 160 mm = 87% of stall stripped a gearbox. This guard "
              "exists because of that.)")
        return 6

    ctrl = Xl330PyController(args.port, args.baudrate, 0.05)
    assert ctrl.ping(args.motor_id), f"id={args.motor_id} not responding"

    # Hardware error bit 0 is "input voltage". On the robot battery (8.4 V vs a
    # 6.0 V rated max) that bit is ALWAYS set and is expected -- it is not in the
    # shutdown mask, so it flags without disabling torque. Do NOT reboot for it:
    # reboot re-initialises the position counter, which combined with a non-zero
    # homing_offset can leave present_position outside the single-turn range.
    hw = int(scalar(ctrl.read_hardware_error_status(args.motor_id)))
    other = hw & ~0x01
    if hw & 0x01:
        print(f"  note: input-voltage error bit set (expected at {'battery' if args.vin is None else 'this'} voltage)")
    if other:
        print(f"  ABORT: hardware_error_status={hw} has non-voltage bits ({other:#04x}); "
              "power-cycle and investigate before recording")
        return 3

    vin = args.vin if args.vin is not None else \
        scalar(ctrl.read_present_input_voltage(args.motor_id)) * 0.1
    temp0 = scalar(ctrl.read_present_temperature(args.motor_id))
    print(f"vin={vin:.2f} V  temp={temp0:.0f} C  kp={args.kp}  "
          f"traj={args.trajectory}  mass={args.mass:.3f}  arm_mass={args.arm_mass:.3f}")

    ctrl.write_torque_enable(args.motor_id, False)
    ctrl.write_operating_mode(args.motor_id, 3)
    ctrl.write_position_p_gain(args.motor_id, args.kp)
    ctrl.write_position_i_gain(args.motor_id, 0)
    ctrl.write_position_d_gain(args.motor_id, 0)
    kp_back = scalar(ctrl.read_position_p_gain(args.motor_id))
    if int(kp_back) != args.kp:
        print(f"  ABORT: kp readback {kp_back:.0f} != {args.kp} (firmware clamped)")
        return 2

    # Sanity-check the position BEFORE energising. A wrapped counter (raw outside
    # 0..4095) or a reading beyond the mechanical limit means enabling torque
    # would command a huge move and drive the arm into the frame.
    q_now = scalar(ctrl.read_present_position(args.motor_id))
    raw_now = scalar(ctrl.read_raw_present_position(args.motor_id))
    print(f"  present position {math.degrees(q_now):+.1f} deg (raw {raw_now:.0f})")
    if not (0.0 <= raw_now <= 4095.0):
        print(f"  ABORT: raw position {raw_now:.0f} outside 0..4095 -- the counter has "
              "wrapped. Power-cycle the servo and re-check before energising.")
        return 4
    if abs(q_now) > abort_at:
        print(f"  ABORT: arm at {math.degrees(q_now):+.1f} deg, outside +-{args.abort_deg:.0f} deg. "
              "Move it back by hand first.")
        return 5

    # Ramp from where it actually is to the trajectory's t=0, so enabling torque
    # cannot snap even if the arm was left displaced.
    q_start, _ = traj(0.0)
    ctrl.write_goal_position(args.motor_id, float(q_now))
    ctrl.write_torque_enable(args.motor_id, True)
    n_pre = 50
    for i in range(n_pre + 1):
        a = 0.5 * (1.0 - math.cos(math.pi * i / n_pre))
        ctrl.write_goal_position(args.motor_id, float(q_now + (q_start - q_now) * a))
        time.sleep(0.02)
    time.sleep(0.3)

    entries = []
    aborted = False
    i_peak = 0.0
    over = 0
    dt = 1.0 / LOG_HZ
    t0 = time.perf_counter()
    torque_on = True
    try:
        while True:
            t = time.perf_counter() - t0
            if t >= duration:
                break
            goal, enable = traj(t)
            goal = float(np.clip(goal, -cmd_limit, cmd_limit))

            if enable != torque_on:
                ctrl.write_torque_enable(args.motor_id, enable)
                torque_on = enable
            if enable:
                ctrl.write_goal_position(args.motor_id, goal)

            q = scalar(ctrl.read_present_position(args.motor_id))
            dq = scalar(ctrl.read_present_velocity(args.motor_id)) * DXL_VEL_TICK_TO_RAD_S

            # Sample current every 5th step: enough for an overload guard and a
            # sparse current channel, without costing much loop rate.
            cur = None
            if len(entries) % 2 == 0:
                cur = scalar(ctrl.read_present_current(args.motor_id)) * 0.001
                i_peak = max(i_peak, abs(cur))
                over = over + 1 if abs(cur) > args.max_current else 0
                if over >= 2:
                    print(f"  ABORT: current {abs(cur):.2f} A > {args.max_current:.2f} A "
                          f"sustained (~{abs(cur)*KT_NM_PER_A/XL330_STALL_NM*100:.0f}% of "
                          "rated stall) -- releasing torque")
                    aborted = True
                    break
            if abs(q) > abort_at:
                print(f"  ABORT: achieved {math.degrees(q):+.1f} deg exceeds "
                      f"+-{args.abort_deg:.0f} deg -- releasing torque, NOT driving back "
                      "(move it by hand)")
                aborted = True
                break
            ent = {"timestamp": t, "position": q, "speed": dq,
                   "goal_position": goal, "torque_enable": bool(enable)}
            if cur is not None:
                ent["current"] = cur
            entries.append(ent)
            sleep = dt - (time.perf_counter() - t0 - t)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\ninterrupted")
        aborted = True
    finally:
        # Always leave the arm hanging at 0, but only DRIVE it there if the run
        # ended cleanly -- after an abort the position reading is not trusted.
        if not aborted and entries:
            q_end = scalar(ctrl.read_present_position(args.motor_id))
            if abs(q_end) <= abort_at:
                ctrl.write_torque_enable(args.motor_id, True)
                n_ret = 75
                for i in range(n_ret + 1):
                    a = 0.5 * (1.0 - math.cos(math.pi * i / n_ret))
                    ctrl.write_goal_position(
                        args.motor_id,
                        float(np.clip(q_end * (1.0 - a), -cmd_limit, cmd_limit)))
                    time.sleep(0.02)
                time.sleep(0.3)
                q_home = scalar(ctrl.read_present_position(args.motor_id))
                print(f"  returned to {math.degrees(q_home):+.2f} deg")
        ctrl.write_torque_enable(args.motor_id, False)

    # bam/process.py interpolates EVERY key it finds in an entry against the
    # next entry, so a key present on only some entries raises KeyError. Current
    # is sampled every 2nd step for loop-rate reasons, so fill it on all entries
    # by linear interpolation before writing.
    if entries:
        have = [(e["timestamp"], e["current"]) for e in entries if "current" in e]
        if have:
            ts = np.array([h[0] for h in have])
            cs = np.array([h[1] for h in have])
            for e in entries:
                e["current"] = float(np.interp(e["timestamp"], ts, cs))
        else:
            for e in entries:
                e["current"] = 0.0

    temp1 = scalar(ctrl.read_present_temperature(args.motor_id))
    qs = np.degrees([e["position"] for e in entries])
    print(f"  {len(entries)} samples, {entries[-1]['timestamp']:.2f} s, "
          f"achieved {qs.min():+.1f} .. {qs.max():+.1f} deg, temp {temp0:.0f}->{temp1:.0f} C, "
          f"peak {i_peak:.2f} A ({i_peak*KT_NM_PER_A/XL330_STALL_NM*100:.0f}% stall)")

    os.makedirs(args.out, exist_ok=True)
    log = {"mass": args.mass, "arm_mass": args.arm_mass, "length": args.length,
           "kp": args.kp, "vin": vin, "trajectory": args.trajectory,
           "entries": entries}
    name = (f"{args.trajectory}_m{int(args.mass*1000)}_kp{args.kp}"
            f"_v{int(round(vin*10))}.json")
    path = os.path.join(args.out, name)
    json.dump(log, open(path, "w"))
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
