"""
Autonomous Flight Controller for AI Grand Prix
=================================================
Vision-based navigation using attitude rate control.
No dependency on position data — navigates purely from:
  - FPV camera (gate detection)
  - ATTITUDE telemetry (roll, pitch, yaw)
  - Race status (active gate index)

Control: SET_ATTITUDE_TARGET (body rates + thrust)
Navigation: Pixel-error-based steering — like an FPV pilot.

State machine: ARMED_WAIT → LIFTOFF → NAVIGATE → FINISHED
"""

import time
import math
import numpy as np
from pymavlink import mavutil

# --------------------------------------------------------------------------------------
# RESET COMMAND
MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------
STATE_ARMED_WAIT = "ARMED_WAIT"      # brief wait after arming
STATE_LIFTOFF    = "LIFTOFF"         # timed ascent
STATE_NAVIGATE   = "NAVIGATE"        # main flight
STATE_FINISHED   = "FINISHED"        # race done

# --------------------------------------------------------------------------------------
# Control Mode — switch if needed
# "attitude" = SET_ATTITUDE_TARGET (body rates + thrust) — most reliable
# "velocity" = SET_POSITION_TARGET_LOCAL_NED (velocity commands) — needs sim FC support
# --------------------------------------------------------------------------------------
CONTROL_MODE = "attitude"

# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
CONTROL_HZ = 50

# Liftoff: duration in seconds to send upward commands
LIFTOFF_DURATION = 2.5   # seconds of upward thrust before navigating
ARMED_WAIT_TIME  = 1.5   # seconds to wait after arm command

# --------------------------------------------------------------------------------------
# Attitude Control Tuning
# --------------------------------------------------------------------------------------
HOVER_THRUST     = 0.52   # thrust for ~hover (adjust per drone weight)
LIFTOFF_THRUST   = 0.70   # thrust during liftoff (more than hover to ascend)
FORWARD_PITCH    = -0.12  # base forward pitch angle in radians (~7°)
MAX_PITCH        = -0.30  # max forward pitch (~17°)
MAX_ROLL         = 0.25   # max roll angle (~14°)

# Outer-loop P gains: desired_angle → rate command
KP_PITCH = 3.5
KP_ROLL  = 3.5

# --------------------------------------------------------------------------------------
# Vision Steering Gains
# --------------------------------------------------------------------------------------
# Pixel error is in range roughly [-320, 320] horizontal, [-180, 180] vertical
KP_YAW_PIXEL    = 0.004    # yaw rate per pixel of horizontal error
KP_ROLL_PIXEL   = 0.0006   # desired roll angle per pixel of horizontal error
KP_THRUST_PIXEL = 0.0006   # thrust adjustment per pixel of vertical error
KP_PITCH_PIXEL  = 0.0002   # pitch adjustment per pixel of vertical error

# --------------------------------------------------------------------------------------
# Speed modulation (via pitch angle)
# --------------------------------------------------------------------------------------
# When gate is very close, reduce forward pitch to slow down
SLOW_DOWN_DISTANCE = 4.0
SLOW_DOWN_FACTOR   = 0.5

# --------------------------------------------------------------------------------------
# Search behavior (no gate visible)
# --------------------------------------------------------------------------------------
SEARCH_PITCH      = -0.08   # gentle forward pitch
SEARCH_YAW_RATE   = 0.4     # yaw rate for searching (rad/s)
SEARCH_THRUST     = 0.52
NO_GATE_TIMEOUT   = 3.0     # after this many seconds with no gate, start searching

# --------------------------------------------------------------------------------------
# Velocity control tuning (if CONTROL_MODE = "velocity")
# --------------------------------------------------------------------------------------
CRUISE_SPEED    = 5.0   # m/s forward
APPROACH_SPEED  = 3.0
PRECISION_SPEED = 2.0
SEARCH_SPEED    = 2.5

# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------
RATES_ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
)

VELOCITY_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms

        # State machine
        self.state = STATE_ARMED_WAIT
        self.state_start_time = time.time()

        # Vision tracking
        self.last_gate_seen_time = 0        # wall-clock time when gate last detected
        self.last_gate_px_err = (0.0, 0.0)  # last known pixel error for momentum
        self.search_direction = 1.0          # +1 or -1 for yaw search direction

        # Gate tracking
        self.last_active_gate = 0
        self.gates_passed = 0

        # Status printing
        self.last_print_time = 0
        self.loop_count = 0

        print(f"[CTRL] Controller initialized. Control mode: {CONTROL_MODE}", flush=True)

    def update(self):
        """Main control tick."""
        self.loop_count += 1
        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == STATE_ARMED_WAIT:
            if elapsed > ARMED_WAIT_TIME:
                print("[CTRL] Arm wait complete.", flush=True)
                self._change_state(STATE_LIFTOFF)
            # Do nothing while waiting

        elif self.state == STATE_LIFTOFF:
            self._handle_liftoff(elapsed)

        elif self.state == STATE_NAVIGATE:
            self._handle_navigate()

        elif self.state == STATE_FINISHED:
            self._handle_finished()

        # Periodic status
        if now - self.last_print_time > 3.0:
            self._print_status()
            self.last_print_time = now

        time.sleep(1.0 / CONTROL_HZ)

    # ===================================================================
    # STATE HANDLERS
    # ===================================================================

    def _handle_liftoff(self, elapsed):
        """Timed ascent — no position check needed."""
        if elapsed > LIFTOFF_DURATION:
            print(f"[CTRL] Liftoff complete ({LIFTOFF_DURATION}s). Starting navigation!", flush=True)
            self._change_state(STATE_NAVIGATE)
            return

        if CONTROL_MODE == "attitude":
            # Thrust up, keep level
            att = self.data.get('attitude', {})
            pitch = att.get('pitch', 0.0)
            roll = att.get('roll', 0.0)

            # Level corrections
            pitch_rate = KP_PITCH * (0.0 - pitch)  # target level
            roll_rate = KP_ROLL * (0.0 - roll)

            self._send_attitude(roll_rate, pitch_rate, 0.0, LIFTOFF_THRUST)
        else:
            # Velocity: go up
            self._send_velocity_ned(0.0, 0.0, -1.5, 0.0)

    def _handle_navigate(self):
        """
        Main navigation — steer toward gates using vision.

        When gate visible: steer toward it.
        When gate NOT visible: fly forward + search yaw.
        """
        # Check race status
        race = self.data.get('race_status', {})
        if race.get('race_finished', False):
            self._change_state(STATE_FINISHED)
            return

        # Track gate progress
        active_gate = race.get('active_gate_index', 0)
        if active_gate > self.last_active_gate:
            self.gates_passed += 1
            self.last_active_gate = active_gate

        # Get current attitude
        att = self.data.get('attitude', {})
        pitch = att.get('pitch', 0.0)
        roll = att.get('roll', 0.0)
        yaw = att.get('yaw', 0.0)

        # Get vision detection
        vision = self.data.get('vision_detection', {})
        detection_time = vision.get('timestamp', 0)
        is_fresh = (time.time() - detection_time) < 0.5  # less than 500ms old
        gate_visible = vision.get('detected', False) and is_fresh

        now = time.time()

        if gate_visible:
            self.last_gate_seen_time = now
            px_err = vision.get('pixel_error', (0.0, 0.0))
            self.last_gate_px_err = px_err
            gate_area = vision.get('gate_area', 0)
            distance = vision.get('distance', None)

            if CONTROL_MODE == "attitude":
                self._navigate_attitude_gate_visible(pitch, roll, yaw, px_err, gate_area, distance)
            else:
                self._navigate_velocity_gate_visible(yaw, px_err, gate_area, distance)
        else:
            time_since_gate = now - self.last_gate_seen_time

            if CONTROL_MODE == "attitude":
                self._navigate_attitude_no_gate(pitch, roll, yaw, time_since_gate)
            else:
                self._navigate_velocity_no_gate(yaw, time_since_gate)

    def _handle_finished(self):
        """Race done — hover."""
        if CONTROL_MODE == "attitude":
            att = self.data.get('attitude', {})
            pitch_rate = KP_PITCH * (0.0 - att.get('pitch', 0.0))
            roll_rate = KP_ROLL * (0.0 - att.get('roll', 0.0))
            self._send_attitude(roll_rate, pitch_rate, 0.0, HOVER_THRUST)
        else:
            self._send_velocity_ned(0.0, 0.0, 0.0, 0.0)

    # ===================================================================
    # ATTITUDE CONTROL NAVIGATION
    # ===================================================================

    def _navigate_attitude_gate_visible(self, pitch, roll, yaw, px_err, gate_area, distance):
        """
        Gate is visible — steer toward it using attitude control.

        px_err = (horizontal_error, vertical_error) in pixels
          Positive horizontal = gate is to the RIGHT of image center
          Positive vertical = gate is BELOW image center
        """
        ex, ey = px_err

        # --- YAW RATE ---
        # Turn toward the gate horizontally
        yaw_rate = KP_YAW_PIXEL * ex
        yaw_rate = clamp(yaw_rate, -1.5, 1.5)

        # --- DESIRED ROLL ---
        # Slight roll into the turn for lateral correction
        desired_roll = KP_ROLL_PIXEL * ex
        desired_roll = clamp(desired_roll, -MAX_ROLL, MAX_ROLL)

        # --- DESIRED PITCH ---
        # Base forward pitch + slight correction from vertical error
        # If gate is below center (positive ey), we might be too high → pitch forward more
        # If gate is above center (negative ey), we might be too low → pitch less forward
        desired_pitch = FORWARD_PITCH - KP_PITCH_PIXEL * ey
        desired_pitch = clamp(desired_pitch, MAX_PITCH, 0.0)

        # Slow down when very close to gate
        if distance is not None and distance < SLOW_DOWN_DISTANCE:
            slowdown = SLOW_DOWN_FACTOR + (1.0 - SLOW_DOWN_FACTOR) * (distance / SLOW_DOWN_DISTANCE)
            desired_pitch *= slowdown

        # --- THRUST ---
        # Base hover + vertical correction
        # Gate above center (negative ey) → need more thrust to go up
        # Gate below center (positive ey) → need less thrust
        thrust = HOVER_THRUST - KP_THRUST_PIXEL * ey
        thrust = clamp(thrust, 0.35, 0.75)

        # --- CONVERT DESIRED ANGLES → RATE COMMANDS ---
        pitch_rate = KP_PITCH * (desired_pitch - pitch)
        roll_rate = KP_ROLL * (desired_roll - roll)

        pitch_rate = clamp(pitch_rate, -2.0, 2.0)
        roll_rate = clamp(roll_rate, -2.0, 2.0)

        self._send_attitude(roll_rate, pitch_rate, yaw_rate, thrust)

    def _navigate_attitude_no_gate(self, pitch, roll, yaw, time_since_gate):
        """
        No gate visible — fly forward and search.
        
        Strategy:
        - Brief momentum: continue last known direction for ~1s
        - Then: fly forward with gentle yaw sweep to find next gate
        """
        if time_since_gate < 1.0 and self.last_gate_seen_time > 0:
            # Momentum phase: keep flying in last known direction briefly
            ex, ey = self.last_gate_px_err
            # Reduced corrections (fading)
            fade = 1.0 - time_since_gate
            yaw_rate = KP_YAW_PIXEL * ex * fade * 0.5
            desired_roll = KP_ROLL_PIXEL * ex * fade * 0.3
            desired_pitch = FORWARD_PITCH
            thrust = HOVER_THRUST - KP_THRUST_PIXEL * ey * fade * 0.3
        else:
            # Search phase: fly forward with yaw sweep
            desired_pitch = SEARCH_PITCH
            desired_roll = 0.0
            thrust = SEARCH_THRUST

            # Yaw search — sweep back and forth
            if time_since_gate > NO_GATE_TIMEOUT:
                # More aggressive search
                yaw_rate = SEARCH_YAW_RATE * self.search_direction * 1.5
                # Flip direction every 3 seconds
                if int(time_since_gate) % 6 < 3:
                    self.search_direction = 1.0
                else:
                    self.search_direction = -1.0
            else:
                yaw_rate = SEARCH_YAW_RATE * self.search_direction * 0.5

        desired_pitch = clamp(desired_pitch, MAX_PITCH, 0.0)
        desired_roll = clamp(desired_roll, -MAX_ROLL, MAX_ROLL)
        thrust = clamp(thrust, 0.35, 0.70)

        pitch_rate = KP_PITCH * (desired_pitch - pitch)
        roll_rate = KP_ROLL * (desired_roll - roll)

        pitch_rate = clamp(pitch_rate, -2.0, 2.0)
        roll_rate = clamp(roll_rate, -2.0, 2.0)
        yaw_rate = clamp(yaw_rate, -1.5, 1.5)

        self._send_attitude(roll_rate, pitch_rate, yaw_rate, thrust)

    # ===================================================================
    # VELOCITY CONTROL NAVIGATION (alternative mode)
    # ===================================================================

    def _navigate_velocity_gate_visible(self, yaw, px_err, gate_area, distance):
        """Gate visible — fly toward it using velocity commands."""
        ex, ey = px_err
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Speed based on distance
        if distance is not None:
            if distance > 10:
                speed = CRUISE_SPEED
            elif distance > 5:
                speed = APPROACH_SPEED
            else:
                speed = PRECISION_SPEED
        else:
            speed = APPROACH_SPEED

        # Lateral and vertical corrections (in body frame)
        lateral = clamp(ex * 0.008, -2.0, 2.0)    # body-right
        vertical = clamp(ey * 0.005, -1.5, 1.5)    # body-down

        # Body velocities
        vx_body = speed          # forward
        vy_body = lateral        # right
        vz_body = vertical       # down

        # Transform body → NED
        vx_ned = cos_y * vx_body - sin_y * vy_body
        vy_ned = sin_y * vx_body + cos_y * vy_body
        vz_ned = vz_body

        # Yaw toward gate
        yaw_rate = clamp(ex * KP_YAW_PIXEL, -1.5, 1.5)

        self._send_velocity_ned(vx_ned, vy_ned, vz_ned, yaw_rate)

    def _navigate_velocity_no_gate(self, yaw, time_since_gate):
        """No gate visible — fly forward and search."""
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        speed = SEARCH_SPEED
        vx_ned = cos_y * speed
        vy_ned = sin_y * speed

        if time_since_gate > NO_GATE_TIMEOUT:
            yaw_rate = SEARCH_YAW_RATE * self.search_direction
            if int(time_since_gate) % 6 < 3:
                self.search_direction = 1.0
            else:
                self.search_direction = -1.0
        else:
            yaw_rate = 0.2 * self.search_direction

        self._send_velocity_ned(vx_ned, vy_ned, 0.0, yaw_rate)

    # ===================================================================
    # COMMAND SENDERS
    # ===================================================================

    def _send_attitude(self, roll_rate, pitch_rate, yaw_rate, thrust):
        """Send attitude rate + thrust command."""
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            RATES_ATTITUDE_MASK,
            [1, 0, 0, 0],      # dummy quaternion (ignored)
            float(roll_rate),
            float(pitch_rate),
            float(yaw_rate),
            float(thrust)
        )

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate):
        """Send velocity command in NED frame."""
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_position_target_local_ned_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            VELOCITY_ONLY_MASK,
            0.0, 0.0, 0.0,
            float(vx), float(vy), float(vz),
            0.0, 0.0, 0.0,
            0.0,
            float(yaw_rate)
        )

    # ===================================================================
    # HELPERS
    # ===================================================================

    def _change_state(self, new_state):
        old = self.state
        self.state = new_state
        self.state_start_time = time.time()
        print(f"[CTRL] State: {old} → {new_state}", flush=True)

    def _print_status(self):
        att = self.data.get('attitude', {})
        race = self.data.get('race_status', {})
        track = self.data.get('track', {})

        pitch_deg = math.degrees(att.get('pitch', 0))
        roll_deg = math.degrees(att.get('roll', 0))
        yaw_deg = math.degrees(att.get('yaw', 0))

        active_gate = race.get('active_gate_index', 0)
        num_gates = track.get('num_gates', '?')

        vision = self.data.get('vision_detection', {})
        det_time = vision.get('timestamp', 0)
        is_fresh = (time.time() - det_time) < 0.5
        gate_vis = vision.get('detected', False) and is_fresh

        vis_str = "NO"
        if gate_vis:
            d = vision.get('distance', 0)
            px = vision.get('pixel_error', (0, 0))
            vis_str = f"YES(d={d:.1f}m px=({px[0]:.0f},{px[1]:.0f}))"

        since_gate = time.time() - self.last_gate_seen_time if self.last_gate_seen_time > 0 else 999

        status = (f"[STATUS] state={self.state} gate={active_gate}/{num_gates} "
                  f"pitch={pitch_deg:.1f}° roll={roll_deg:.1f}° yaw={yaw_deg:.0f}° "
                  f"vision={vis_str} last_seen={since_gate:.1f}s ago")

        print(status, flush=True)

    # ===================================================================
    # ARM / RESET
    # ===================================================================

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0, 0, 0, 0, 0, 0, 0, 0
        )
