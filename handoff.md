# AI-GP Simulator Autonomous Drone — Handoff Document

## Overview

Autonomous drone racing controller for **AI Grand Prix Simulator v1.0.3364**. Python + MAVLink (pymavlink) controls ArduPilot inside the Unity simulator.

Navigation:
1. **Waypoint navigation** — Gate positions from track data (Z inverted for NED)
2. **Vision refinement** — OpenCV gate detection when within 8m (optional blend)

---

## Architecture

| File | Purpose |
|------|---------|
| `main.py` | Connect, `SIM_RESET`, 3s wait, control loop |
| `controller.py` | WAIT → ARMING → TAKEOFF (timer) → NAVIGATE → FINISHED |
| `setup.py` | Wires MAVLink, vision, controller |
| `mavlink_rx.py` | Telemetry → `shared_data` |
| `vision_rx.py` | FPV frames → gate detector (errors must not crash thread) |
| `gate_detector.py` | HSV + contours + solvePnP |

---

## Coordinate System (CRITICAL)

**Flight controller uses NED:** X=North, Y=East, Z=Down (negative Z = up).

**Track gate Z from simulator is altitude-UP**, not true NED — despite `mavlink_rx.py` calling it `position_ned_z`. Gate 1 at raw Z=+5.1 means 5.1m above gate 0, not underground.

**Always invert track Z when loading gates:**
```python
pos = np.array([raw[0], raw[1], -raw[2]])
```

**Aim altitude floor:** `aim_point[2] = min(aim_point[2], -1.0)` so the drone never dives at ground-level gate centers (Z=0 raw).

---

## Startup Sequence (proven)

1. `main.py` sends `SIM_RESET` (cmd 31000), waits **3.0 seconds**
2. Control loop starts; controller **WAIT** state:
   - Mandatory **4.0s** EKF boot delay after script start (`Booting EKF...`)
   - Then when `race_started` **or** `track.received`: force **GUIDED** mode once, load track, arm
3. **ARMING:** spam `arm()` for 0.5s (UDP packet loss)
4. **TAKEOFF:** **timer only** — `vz = -2.5` for **2.5s** — **never** use altitude telemetry here (stays 0,0,0 after reset → infinite climb)
5. **NAVIGATE:** NED velocity toward gate aim point

### DO NOT use

| Anti-pattern | Why |
|--------------|-----|
| `rc_channels_override_send` | Hijacks throttle; FC ignores velocity commands |
| Body-frame velocity hacks | Wrong; use `MAV_FRAME_LOCAL_NED` |
| `set_mode` spam every tick | Unnecessary |
| Attitude/ACRO takeoff | Sticks ACRO mode, breaks velocity control |
| Altitude-based takeoff after SIM_RESET | Position stuck at z=0 → climb forever, no forward |

---

## Navigation

- Aim: `gate_pos + gate_forward * 1.5m`, altitude floor -1.0 NED
- Speed: 6 / 3.5 / 2.5 m/s by distance
- Yaw: P-gain 0.7, clamp ±1.0 rad/s from **`to_aim`** (not vision-adjusted velocity)
- Alignment throttle: `vel_cmd *= max(0.1, cos(yaw_err))` on **all 3 axes**
- Vision blend: ≤30% weight within 8m

---

## Status logging

Every 0.5s: `[STATUS] state armed pos speed yaw cmd=(vx,vy,vz,yaw_rate) gate dist`

After TAKEOFF, `cmd` must show **non-zero vx/vy** toward gate 0 (~-23m North).

---

## Running

1. **Restart FlightSim** (clears stuck ACRO from old tests)
2. Start race in simulator
3. `cd PyAIPilotExample && pip install -r requirements.txt && python main.py`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Flies up only, no forward | Altitude takeoff + stale z=0 | Use timer takeoff (current controller) |
| Dives / spins on navigate | Track Z not inverted | `-raw[2]` in `_load_track` |
| pos jumps to z=-93 | Armed before EKF ready | Wait 4s + 3s reset |
| cmd correct but no motion | RC overrides | Remove them |
| Violent yaw spin | yaw_rate unclamped / vision yaw loop | clamp ±1.0, yaw from `to_aim` only |

---

## Tuning (controller.py)

| Parameter | Value |
|-----------|-------|
| `EKF_BOOT_DELAY_S` | 4.0 |
| `TAKEOFF_DURATION_S` | 2.5 |
| `TAKEOFF_VZ` | -2.5 |
| `MAX_SPEED` | 6.0 |
| `GATE_LOOKAHEAD` | 1.5 |
| `AIM_ALTITUDE_NED` | -1.0 |
