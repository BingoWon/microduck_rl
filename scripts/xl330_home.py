#!/usr/bin/env python3
"""Check an XL330 and optionally drive it to its zero position.

Read-only by default: it pings the motor and dumps its state. Actually moving
the arm requires --move, because with a weighted arm on the bench that is a
physical action.

  # Inspect only (safe, no torque):
  uv run python scripts/xl330_home.py --port /dev/ttyUSB0 --id 15

  # Ramp slowly to 0 rad and hold:
  uv run python scripts/xl330_home.py --port /dev/ttyUSB0 --id 15 --move

  # Ramp to zero, then release so the arm hangs free:
  uv run python scripts/xl330_home.py --port /dev/ttyUSB0 --id 15 --move --release

Zero here is the SERVO's zero (goal_position 0.0 rad), which equals "arm
straight down" only if the arm was mounted that way. The holding current
reported at the end tells you: near zero means the servo is not fighting
gravity, i.e. zero really is the hanging equilibrium. A few hundred mA means
there is a mounting offset -- see --homing-offset-hint.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from rustypot import Xl330PyController

# Conservative gain for the homing move. The bench's identification runs use
# much stiffer values; this is only meant to walk the arm over gently.
SAFE_KP = 200
RAD_PER_TICK = 2.0 * math.pi / 4096.0


def scalar(x) -> float:
    """rustypot returns either a scalar or a 1-element list depending on call."""
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


def try_read(ctrl, name: str, motor_id: int):
    """Read a register, returning None if this firmware/binding lacks it."""
    fn = getattr(ctrl, name, None)
    if fn is None:
        return None
    try:
        return scalar(fn(motor_id))
    except Exception as exc:  # noqa: BLE001 - diagnostic tool, report and continue
        print(f"    ({name} failed: {exc})")
        return None


def report(ctrl, motor_id: int, header: str) -> dict:
    print(f"\n{header}")
    vals = {}
    for label, reg, fmt in [
        ("model number", "read_model_number", "{:.0f}"),
        ("firmware", "read_firmware_version", "{:.0f}"),
        ("operating mode", "read_operating_mode", "{:.0f}"),
        ("torque enable", "read_torque_enable", "{:.0f}"),
        ("position P gain", "read_position_p_gain", "{:.0f}"),
        ("temperature (C)", "read_present_temperature", "{:.1f}"),
        ("input voltage (V)", "read_present_input_voltage", "{:.2f}"),
        ("current (A)", "read_present_current", "{:.3f}"),
        ("current limit", "read_current_limit", "{:.0f}"),
        ("homing offset", "read_homing_offset", "{:.4f}"),
    ]:
        v = try_read(ctrl, reg, motor_id)
        vals[reg] = v
        if v is None:
            continue
        # present_current comes back in mA on this binding.
        if reg == "read_present_current":
            v = v * 0.001
        print(f"    {label:<20} {fmt.format(v)}")

    q = try_read(ctrl, "read_present_position", motor_id)
    raw = try_read(ctrl, "read_raw_present_position", motor_id)
    if q is not None:
        vals["q"] = q
        raw_s = f"  (raw {raw:.0f})" if raw is not None else ""
        print(f"    {'position':<20} {q:+.4f} rad = {math.degrees(q):+7.2f} deg{raw_s}")
    return vals



# --- configuration for BAM identification runs --------------------------------
#
# What each register has to be, and why:
#   position_i/d_gain, feedforward_*  = 0   BAM models a pure-P firmware law.
#                                           Any I/D/FF term is unmodelled and
#                                           gets absorbed into fitted friction.
#   acceleration_limit               = 0    0 means "no profile" -> raw position
#                                           control. Non-zero makes the servo
#                                           interpolate, which is not what BAM
#                                           simulates.
#   velocity_limit                   = max  The stock 445 (10.7 rad/s) clamps
#                                           BELOW the motor's own no-load speed
#                                           (~14 rad/s at 5.2 V), so the firmware
#                                           would bind before back-EMF does and
#                                           the fit would blame friction.
#   min/max_position_limit           = wide Three of BAM's five identification
#                                           trajectories reach +-90..119 deg.
#                                           Clipping biases the friction terms.
VELOCITY_LIMIT_MAX = 2047  # XL330 register max (~49 rad/s)



def zero_here(ctrl, motor_id: int) -> bool:
    """Make the arm's present position read zero.

    Iterative rather than computed: rustypot's homing_offset has no `raw`
    variant and returns ticks while positions come back in radians, so rather
    than assume the write unit we nudge and re-read until it converges. Two or
    three passes is enough either way.
    """
    print("\n--- zeroing here ---")
    ctrl.write_torque_enable(motor_id, False)
    time.sleep(0.1)
    q = try_read(ctrl, "read_present_position", motor_id)
    off = try_read(ctrl, "read_homing_offset", motor_id)
    if q is None or off is None:
        print("    could not read position/offset")
        return False
    print(f"    start: position {math.degrees(q):+.2f} deg, homing_offset {off:.0f}")

    for it in range(6):
        q = try_read(ctrl, "read_present_position", motor_id)
        if q is None:
            return False
        if abs(math.degrees(q)) < 0.1:
            print(f"    converged after {it} write(s): {math.degrees(q):+.3f} deg")
            break
        off = try_read(ctrl, "read_homing_offset", motor_id)
        # ticks per radian, from the 4096-count encoder
        delta = -q * 4096.0 / (2.0 * math.pi)
        new = off + delta
        if not (-1044480 <= new <= 1044480):  # register range
            print(f"    offset {new:.0f} out of range; aborting")
            return False
        ctrl.write_homing_offset(motor_id, int(round(new)))
        time.sleep(0.05)
        q2 = try_read(ctrl, "read_present_position", motor_id)
        print(f"    offset {off:.0f} -> {new:.0f}: position "
              f"{math.degrees(q):+.2f} -> {math.degrees(q2 or 0):+.2f} deg")
    else:
        print("    did NOT converge - check the write unit and the arm is free")
        return False

    raw = try_read(ctrl, "read_raw_present_position", motor_id)
    if raw is not None and not (0.0 <= raw <= 4095.0):
        print(f"    WARNING: raw {raw:.0f} outside 0..4095 after zeroing; "
              "the counter is wrapped and the recorder will refuse to run")
        return False
    return True


def configure(ctrl, motor_id: int, max_deg: float, velocity_limit: int,
              kp: int | None) -> bool:
    """Write the registers the identification runs depend on. Returns True if
    every value read back as requested."""
    print(f"\n--- configuring id={motor_id} for identification ---")
    print("  disabling torque (EEPROM writes are ignored while torque is on)")
    ctrl.write_torque_enable(motor_id, False)
    time.sleep(0.1)

    lim = math.radians(max_deg)
    writes = [
        # (register, value, readback register, tolerance)
        ("write_min_position_limit", -lim, "read_min_position_limit", 2e-3),
        ("write_max_position_limit", lim, "read_max_position_limit", 2e-3),
        ("write_velocity_limit", velocity_limit, "read_velocity_limit", 0.5),
        ("write_operating_mode", 3, "read_operating_mode", 0.5),
        ("write_position_i_gain", 0, "read_position_i_gain", 0.5),
        ("write_position_d_gain", 0, "read_position_d_gain", 0.5),
        ("write_feedforward_1st_gain", 0, "read_feedforward_1st_gain", 0.5),
        ("write_feedforward_2nd_gain", 0, "read_feedforward_2nd_gain", 0.5),
    ]
    if kp is not None:
        writes.append(("write_position_p_gain", kp, "read_position_p_gain", 0.5))

    ok = True
    for wname, value, rname, tol in writes:
        wfn = getattr(ctrl, wname, None)
        if wfn is None:
            print(f"    {wname:<28} NO SUCH REGISTER in this rustypot build")
            ok = False
            continue
        try:
            wfn(motor_id, value)
        except Exception as exc:  # noqa: BLE001
            print(f"    {wname:<28} WRITE FAILED: {exc}")
            ok = False
            continue
        time.sleep(0.02)
        back = try_read(ctrl, rname, motor_id)
        if back is None:
            print(f"    {wname:<28} wrote {value}, could not read back")
            ok = False
            continue
        good = abs(back - float(value)) <= tol
        flag = "ok" if good else "MISMATCH (firmware clamped?)"
        print(f"    {wname:<28} wrote {float(value):>10.4f}  read {back:>10.4f}  {flag}")
        ok = ok and good

    # acceleration_limit must be 0 (no trapezoidal profile). Only report it --
    # writing it is not always permitted and 0 is the stock value.
    acc = try_read(ctrl, "read_acceleration_limit", motor_id)
    if acc is not None and acc != 0.0:
        print(f"    acceleration_limit is {acc:.0f}, expected 0 "
              "(non-zero = the servo profiles the move, which BAM does not model)")
        ok = False

    print("\n  configuration " + ("OK" if ok else "INCOMPLETE - see mismatches above"))
    print("  torque left DISABLED.")
    if ok:
        print(f"\n  NOTE: software limits are now +-{max_deg:.0f} deg. Confirm the "
              "MECHANISM allows that\n        before driving to the extremes -- "
              "widening the limits does not add clearance.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--id", type=int, default=15, dest="motor_id")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--move", action="store_true",
                    help="actually drive to the goal (default: read-only)")
    ap.add_argument("--goal-deg", type=float, default=0.0,
                    help="target in degrees (default 0 = servo zero)")
    ap.add_argument("--ramp-time", type=float, default=3.0,
                    help="seconds to ramp from present position to goal")
    ap.add_argument("--kp", type=int, default=SAFE_KP,
                    help=f"position P gain for the move (default {SAFE_KP})")
    ap.add_argument("--release", action="store_true",
                    help="disable torque once the goal is reached")
    ap.add_argument("--homing-offset", type=int, default=None,
                    help="write homing_offset directly (ticks). Use 0 for the "
                         "servo's native frame, i.e. raw 2048 == 0 deg.")
    ap.add_argument("--zero-here", action="store_true",
                    help="set homing_offset so the CURRENT position reads 0. "
                         "Arm must be mounted and hanging free with torque off.")
    ap.add_argument("--configure", action="store_true",
                    help="write the registers the BAM identification runs need")
    ap.add_argument("--max-deg", type=float, default=120.0,
                    help="software position limit, +-deg (default 120)")
    ap.add_argument("--velocity-limit", type=int, default=VELOCITY_LIMIT_MAX,
                    help=f"velocity limit register (default {VELOCITY_LIMIT_MAX} = max)")
    args = ap.parse_args()

    print(f"opening {args.port} @ {args.baudrate} baud, id={args.motor_id}")
    ctrl = Xl330PyController(args.port, args.baudrate, 0.05)

    if not ctrl.ping(args.motor_id):
        print(f"ERROR: motor id={args.motor_id} not responding on {args.port}")
        print("  check: power on? correct id? correct port? correct baudrate?")
        return 1
    print("ping OK")

    before = report(ctrl, args.motor_id, "--- state before ---")

    if args.homing_offset is not None:
        print(f"\n--- writing homing_offset = {args.homing_offset} ---")
        ctrl.write_torque_enable(args.motor_id, False)
        time.sleep(0.1)
        ctrl.write_homing_offset(args.motor_id, args.homing_offset)
        time.sleep(0.05)
        back = try_read(ctrl, "read_homing_offset", args.motor_id)
        raw = try_read(ctrl, "read_raw_present_position", args.motor_id)
        q = try_read(ctrl, "read_present_position", args.motor_id)
        print(f"    readback {back:.0f}; position now {math.degrees(q or 0):+.2f} deg "
              f"(raw {raw:.0f})")
        if back is None or int(back) != args.homing_offset:
            print("    MISMATCH - offset did not land")
            return 8
        if raw is not None and not (0.0 <= raw <= 4095.0):
            print("    WARNING: raw outside 0..4095")

    if args.zero_here:
        if not zero_here(ctrl, args.motor_id):
            return 7
        report(ctrl, args.motor_id, "--- state after zeroing ---")

    if args.configure:
        kp = args.kp if any(a.startswith("--kp") for a in sys.argv[1:]) else None
        if not configure(ctrl, args.motor_id, args.max_deg,
                         args.velocity_limit, kp):
            return 2
        report(ctrl, args.motor_id, "--- state after configure ---")

    if not args.move:
        print("\nRead-only (no --move given). Nothing was energised.")
        return 0

    q0 = before.get("q")
    if q0 is None:
        print("\nERROR: could not read present position; refusing to move blind.")
        return 1

    goal = math.radians(args.goal_deg)
    print(f"\n--- moving {math.degrees(q0):+.2f} deg -> {args.goal_deg:+.2f} deg "
          f"over {args.ramp_time:.1f}s at kp={args.kp} ---")

    ctrl.write_torque_enable(args.motor_id, False)
    ctrl.write_operating_mode(args.motor_id, 3)  # position control
    ctrl.write_position_p_gain(args.motor_id, args.kp)
    ctrl.write_position_i_gain(args.motor_id, 0)
    ctrl.write_position_d_gain(args.motor_id, 0)

    kp_back = try_read(ctrl, "read_position_p_gain", args.motor_id)
    if kp_back is not None and int(kp_back) != args.kp:
        print(f"  NOTE: P gain readback {kp_back:.0f} != requested {args.kp} "
              "(firmware clamps out-of-range values)")

    # Start holding where it already is, so enabling torque does not snap.
    ctrl.write_goal_position(args.motor_id, float(q0))
    ctrl.write_torque_enable(args.motor_id, True)
    time.sleep(0.1)

    steps = max(1, int(args.ramp_time / 0.02))
    try:
        for i in range(steps + 1):
            a = (i / steps)
            # cosine ease-in/out: no step at the start, no overshoot at the end
            s = 0.5 * (1.0 - math.cos(math.pi * a))
            ctrl.write_goal_position(args.motor_id, float(q0 + (goal - q0) * s))
            time.sleep(0.02)
        time.sleep(0.5)  # settle
    except KeyboardInterrupt:
        print("\ninterrupted - disabling torque")
        ctrl.write_torque_enable(args.motor_id, False)
        return 130

    after = report(ctrl, args.motor_id, "--- state after ---")

    q1 = after.get("q")
    cur = after.get("read_present_current")
    if q1 is not None:
        err_deg = math.degrees(q1 - goal)
        print(f"\n    position error vs goal: {err_deg:+.2f} deg")
    if cur is not None:
        a_hold = abs(cur) * 0.001
        print(f"    holding current:        {a_hold:.3f} A")
        if args.goal_deg == 0.0:
            if a_hold < 0.03:
                print("    -> near zero: servo zero IS the gravity equilibrium "
                      "(arm hangs straight down here).")
            else:
                print("    -> non-trivial: the servo is holding against gravity, so "
                      "servo zero is NOT straight down.")
                print("       Let the arm hang free, read the position, and set that "
                      "as homing_offset (or remount the arm).")

    if args.release:
        ctrl.write_torque_enable(args.motor_id, False)
        print("\ntorque disabled - arm is free.")
    else:
        print("\ntorque still ENABLED and holding. Re-run with --release to free it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
