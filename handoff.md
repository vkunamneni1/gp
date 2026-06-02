# AI-GP Simulator Autonomous Drone — Handoff Document

## Overview

This project is an autonomous drone racing controller for the **AI Grand Prix Simulator v1.0.3364**. It uses Python + MAVLink (pymavlink) to control a simulated quadcopter running ArduPilot inside a Unity-based 3D physics environment.

The drone navigates through a series of 3D gates using:
1. **Global waypoint navigation** — Gate positions in NED coordinates from track data
2. **Computer vision refinement** — OpenCV-based gate detection from FPV camera for precision threading

---

## Architecture

### Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point. Connects to sim, sends `SIM_RESET`, runs control loop |
| `controller.py` | State machine: IDLE → WAIT_FOR_DATA → TAKEOFF → NAVIGATE → RECOVER → FINISHED |
| `setup.py` | Wires up all components (MAVLink connection, receivers, controller) |
| `mavlink_rx.py` | Background thread parsing MAVLink UDP telemetry into `shared_data` dict |
| `vision_rx.py` | Background thread receiving FPV JPEG frames, running gate detection |
| `gate_detector.py` | OpenCV gate detection via HSV color segmentation + solvePnP |
| `timesync.py` | MAVLink time synchronization |

### Data Flow

```
Simulator (Unity)
    ├─ UDP 14550 ──→ mavlink_rx.py ──→ shared_data (position, attitude, race_status, track, collision)
    └─ UDP 5600  ──→ vision_rx.py ──→ shared_data['vision_detection']

controller.py reads shared_data, sends MAVLink velocity commands back via UDP 14550
```

### State Machine

```
IDLE (3s EKF wait)
  → WAIT_FOR_DATA (wait for position + track telemetry, then arm)
    → TAKEOFF (ascend to -2.0m NED using velocity, altitude-checked)
      → NAVIGATE (waypoint + vision blend, gate-by-gate)
        → RECOVER (on heavy collision: back up 1s)
          → NAVIGATE
        → FINISHED (race_finished flag → hover)
```

---

## Coordinate System

**All coordinates are NED (North-East-Down):**
- X = North (positive forward)
- Y = East (positive right)  
- Z = Down (positive down, **negative = up**)

The track data from `mavlink_rx.py` is **already in NED** (see line 308: `position_ned_x, position_ned_y, position_ned_z`). **No coordinate inversion is needed.**

> [!CAUTION]
> The previous agent incorrectly assumed the simulator sent Z-up coordinates and added a `-Z` inversion in `_load_track()`. This was **wrong** — the track data comments explicitly say NED. The inversion was removed.

---

## Known Issues & Fixes Applied

### 1. SIM_RESET + EKF Stabilization
- `main.py` sends `SIM_RESET` (cmd 31000) on startup to teleport the drone to the start line
- ArduPilot's EKF needs time to recalibrate after this teleport
- **Fix:** `main.py` waits 3.0s after reset, then the controller's IDLE state waits another 3.0s before transitioning

### 2. RC Failsafe
- The simulator triggers an RC-loss failsafe if it doesn't receive RC channel overrides
- **Fix:** `controller.update()` sends `rc_channels_override_send()` every tick (50 Hz) with centered sticks and mid-throttle

### 3. Flight Mode
- The drone must be in GUIDED mode to accept `SET_POSITION_TARGET_LOCAL_NED` velocity commands
- **Fix:** Controller forces GUIDED mode (custom_mode=4) during IDLE→WAIT_FOR_DATA transition

### 4. UDP Packet Loss on Arm
- Single `arm()` commands can be dropped over UDP
- **Fix:** Controller sends `arm()` 5 times in quick succession, plus continues arming during early takeoff

---

## Tuning Parameters (in controller.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_SPEED` | 6.0 m/s | Cruise speed on long straights |
| `APPROACH_SPEED` | 3.5 m/s | Near gate |
| `PRECISION_SPEED` | 2.5 m/s | Very close to gate |
| `TAKEOFF_ALT` | -2.0 m | Target altitude (NED) |
| `TAKEOFF_SPEED` | 1.0 m/s | Ascent rate |
| `GATE_LOOKAHEAD` | 2.5 m | Aim past gate center for clean pass-through |
| `VISION_BLEND_DISTANCE` | 8.0 m | Start trusting vision within this range |
| `VISION_WEIGHT_MAX` | 0.3 | Max vision correction weight |
| `VISION_PIXEL_GAIN` | 0.003 | Pixel error → velocity multiplier |

---

## Navigation Logic (in `_handle_navigate`)

1. **Aim point calculation:** `gate_pos + gate_forward_2d * GATE_LOOKAHEAD`
2. **Next-gate blending:** When close to current gate, subtly blend aim toward next gate
3. **Speed profile:** Distance-based linear interpolation between MAX/APPROACH/PRECISION speeds
4. **Vision correction:** When gate detected and within blend distance, apply pixel error as lateral/vertical velocity correction in NED frame
5. **Yaw control:** P-controller with gain=2.0, clamped to ±2.0 rad/s
6. **Speed clamping:** Total velocity vector magnitude capped at MAX_SPEED

---

## Gate Detector (gate_detector.py)

- HSV color segmentation with auto-locking (tries all colors, locks after 8 consistent detections)
- Contour analysis with aspect ratio filtering
- Distance estimation via gate area or solvePnP when 4 corners detected
- Camera: 640×360, fx=fy=320, 20° upward tilt

---

## Running

```bash
cd PyAIPilotExample
python main.py
```

Requirements: `pymavlink`, `numpy`, `opencv-python` (see `requirements.txt`)

---

## What Still Needs Testing

1. **Does it actually take off and fly?** — The previous agent's broken Z-inversion and various patches made it impossible to verify. This clean version restores the original proven architecture. Run it and observe.
2. **Gate transitions** — Does the drone smoothly move from gate to gate?
3. **Vision accuracy** — Is the gate detector reliably finding gates in this environment?
4. **Speed tuning** — Once gates are being cleared, optimize speeds for better lap times.
