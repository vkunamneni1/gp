"""
Autonomous Flight Controller for AI Grand Prix — v3
=====================================================
FIXES from v2: Proper arming sequence (throttle-down before arm).
Uses velocity commands (SET_POSITION_TARGET_LOCAL_NED) since
LOCAL_POSITION_NED IS available. Waypoint navigation + vision refinement.

State machine:
  THROTTLE_DOWN → ARMING → WAIT_ARM → LIFTOFF → NAVIGATE → FINISHED

Control: velocity commands via SET_POSITION_TARGET_LOCAL_NED
Fallback: attitude rate commands via SET_ATTITUDE_TARGET
"""

import time
import math
import numpy as np
from pymavlink import mavutil

# --------------------------------------------------------------------------------------
MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------
STATE_THROTTLE_DOWN = "THROTTLE_DOWN"  # send zero-throttle before arming
STATE_ARMING        = "ARMING"         # send arm command
STATE_WAIT_ARM      = "WAIT_ARM"       # wait for arm confirmation
STATE_LIFTOFF       = "LIFTOFF"        # ascend to flight altitude
STATE_NAVIGATE      = "NAVIGATE"       # main flight
STATE_FINISHED      = "FINISHED"       # race complete

# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
CONTROL_HZ = 50
THROTTLE_DOWN_DURATION = 3.0   # seconds to send zero-thrust before arm
ARM_WAIT_TIMEOUT       = 5.0   # max seconds to wait for arm confirmation
LIFTOFF_DURATION       = 3.0   # seconds of upward velocity
LIFTOFF_VZ             = -2.0  # upward velocity in NED (negative = up)

# --------------------------------------------------------------------------------------
# Navigation Tuning
# --------------------------------------------------------------------------------------
MAX_SPEED       = 6.0     # m/s cruise
APPROACH_SPEED  = 3.5     # m/s approaching gate
PRECISION_SPEED = 2.5     # m/s close to gate
SEARCH_SPEED    = 2.5     # m/s when no gate visible

FAR_DISTANCE       = 12.0
APPROACH_DISTANCE  = 6.0
PRECISION_DISTANCE = 3.0

GATE_LOOKAHEAD = 2.5      # aim this far past gate center

# Vision refinement
VISION_BLEND_DISTANCE = 8.0
VISION_PIXEL_GAIN     = 0.003
VISION_WEIGHT_MAX     = 0.3

# Search behavior
SEARCH_YAW_RATE    = 0.5   # rad/s yaw when searching
NO_GATE_TIMEOUT    = 3.0   # seconds before aggressive search

# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------
VELOCITY_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

RATES_ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def quat_forward_vector(qw, qx, qy, qz):
    """Get forward direction (gate normal) from orientation quaternion."""
    fx = 1.0 - 2.0*(qy*qy + qz*qz)
    fy = 2.0*(qx*qy + qw*qz)
    fz = 2.0*(qx*qz - qw*qy)
    return np.array([fx, fy, fz])


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms

        # State
        self.state = STATE_THROTTLE_DOWN
        self.state_start_time = time.time()

        # Navigation
        self.target_gate_index = 0
        self.gates = []
        self.num_gates = 0

        # Vision
        self.last_gate_seen_time = 0
        self.search_direction = 1.0

        # Status
        self.last_print_time = 0
        self.loop_count = 0

        print(f"[CTRL] Controller initialized. Sending zero-throttle for arming...", flush=True)

    def update(self):
        """Main control tick."""
        self.loop_count += 1
        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == STATE_THROTTLE_DOWN:
            self._handle_throttle_down(elapsed)

        elif self.state == STATE_ARMING:
            self._handle_arming()

        elif self.state == STATE_WAIT_ARM:
            self._handle_wait_arm(elapsed)

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

    def _handle_throttle_down(self, elapsed):
        """
        Send zero-throttle commands to satisfy the FC's arming check.
        The sim says "throttle down please" if you try to arm with non-zero thrust.
        """
        # Send zero-thrust attitude command every tick
        self._send_idle()

        if elapsed > THROTTLE_DOWN_DURATION:
            self._change_state(STATE_ARMING)

    def _handle_arming(self):
        """Send arm command and move to wait state."""
        # Keep sending zero thrust
        self._send_idle()

        # Send arm command
        print("[CTRL] Sending ARM command...", flush=True)
        self.arm()
        self._change_state(STATE_WAIT_ARM)

    def _handle_wait_arm(self, elapsed):
        """Wait for arm confirmation. Keep sending zero thrust."""
        # Keep sending zero throttle while waiting
        self._send_idle()

        # Check if armed
        if self.data.get('armed', False):
            print("[CTRL] *** ARMED SUCCESSFULLY! ***", flush=True)

            # Load track data if available
            if self.data.get('track', {}).get('received', False):
                self._load_track_data()

            self._change_state(STATE_LIFTOFF)
            return

        if elapsed > ARM_WAIT_TIMEOUT:
            # Try arming again
            print("[CTRL] Arm timeout — retrying...", flush=True)
            self.arm()
            self.state_start_time = time.time()

    def _handle_liftoff(self, elapsed):
        """Timed ascent using velocity commands."""
        if elapsed > LIFTOFF_DURATION:
            # Check position if available
            pos = self.data.get('local_position', {})
            z = pos.get('z', 0)
            print(f"[CTRL] Liftoff complete. Position z={z:.2f}m. Starting navigation!", flush=True)

            # Re-check track data (might have arrived during liftoff)
            if self.num_gates == 0 and self.data.get('track', {}).get('received', False):
                self._load_track_data()

            self._change_state(STATE_NAVIGATE)
            return

        # Go up!
        self._send_velocity_ned(0.0, 0.0, LIFTOFF_VZ, 0.0)

    def _handle_navigate(self):
        """
        Main navigation: waypoint-based + vision refinement.
        Uses track data (gate positions) + local_position_ned for waypoint nav.
        Uses vision for fine-tuning approach to gate opening.
        """
        # Check race finish
        race = self.data.get('race_status', {})
        if race.get('race_finished', False):
            self._change_state(STATE_FINISHED)
            return

        # Sync target gate from race status
        race_gate = race.get('active_gate_index', 0)
        if race_gate > self.target_gate_index:
            self.target_gate_index = race_gate

        # Get drone state
        pos = self.data.get('local_position', {})
        drone_pos = np.array([pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)])

        att = self.data.get('attitude', {})
        drone_yaw = att.get('yaw', 0.0)

        # Get vision
        vision = self.data.get('vision_detection', {})
        vis_fresh = (time.time() - vision.get('timestamp', 0)) < 0.5
        gate_visible = vision.get('detected', False) and vis_fresh
        if gate_visible:
            self.last_gate_seen_time = time.time()

        # ---------------------------------------------------------------
        # WAYPOINT NAVIGATION (if track data available)
        # ---------------------------------------------------------------
        if self.num_gates > 0 and self.target_gate_index < self.num_gates:
            gate = self.gates[self.target_gate_index]
            gate_pos = np.array(gate['position'])
            gate_ori = gate['orientation']  # (w, x, y, z)

            # Gate forward direction (for look-ahead)
            gate_forward = quat_forward_vector(*gate_ori)
            gate_forward_2d = normalize(np.array([gate_forward[0], gate_forward[1], 0.0]))

            # Aim point: past the gate center so we fly THROUGH it
            aim_point = gate_pos + gate_forward_2d * GATE_LOOKAHEAD

            # If next gate exists, blend aim direction for smooth turns
            if self.target_gate_index + 1 < self.num_gates:
                next_gate_pos = np.array(self.gates[self.target_gate_index + 1]['position'])
                next_dir = normalize(next_gate_pos - gate_pos)
                dist_to_gate = np.linalg.norm(gate_pos - drone_pos)
                if dist_to_gate < APPROACH_DISTANCE:
                    blend = (1.0 - dist_to_gate / APPROACH_DISTANCE) * 0.3
                    aim_point = aim_point + next_dir * blend * 3.0

            # Vector from drone to aim point
            to_target = aim_point - drone_pos
            dist_to_gate = np.linalg.norm(gate_pos - drone_pos)
            direction = normalize(to_target)

            # Speed profile
            if dist_to_gate > FAR_DISTANCE:
                speed = MAX_SPEED
            elif dist_to_gate > APPROACH_DISTANCE:
                t = (dist_to_gate - APPROACH_DISTANCE) / (FAR_DISTANCE - APPROACH_DISTANCE)
                speed = APPROACH_SPEED + t * (MAX_SPEED - APPROACH_SPEED)
            elif dist_to_gate > PRECISION_DISTANCE:
                t = (dist_to_gate - PRECISION_DISTANCE) / (APPROACH_DISTANCE - PRECISION_DISTANCE)
                speed = PRECISION_SPEED + t * (APPROACH_SPEED - PRECISION_SPEED)
            else:
                speed = PRECISION_SPEED

            # Base velocity command
            vel_cmd = direction * speed

            # Vision refinement when close to gate
            if gate_visible and dist_to_gate < VISION_BLEND_DISTANCE:
                px_err = vision.get('pixel_error', (0, 0))
                t = 1.0 - (dist_to_gate / VISION_BLEND_DISTANCE)
                weight = t * VISION_WEIGHT_MAX

                cos_y = math.cos(drone_yaw)
                sin_y = math.sin(drone_yaw)

                # Lateral correction
                lat_corr = px_err[0] * VISION_PIXEL_GAIN * weight
                vel_cmd[0] += -sin_y * lat_corr
                vel_cmd[1] += cos_y * lat_corr

                # Vertical correction
                vert_corr = px_err[1] * VISION_PIXEL_GAIN * weight
                vel_cmd[2] += vert_corr

            # Yaw toward target
            desired_yaw = math.atan2(to_target[1], to_target[0])
            yaw_err = desired_yaw - drone_yaw
            while yaw_err > math.pi: yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi
            yaw_rate = clamp(yaw_err * 2.0, -2.0, 2.0)

            # Clamp speed
            spd = np.linalg.norm(vel_cmd)
            if spd > MAX_SPEED:
                vel_cmd = vel_cmd / spd * MAX_SPEED

            self._send_velocity_ned(vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate)

        # ---------------------------------------------------------------
        # VISION-ONLY NAVIGATION (no track data)
        # ---------------------------------------------------------------
        elif gate_visible:
            self._navigate_vision_only(drone_yaw, vision)

        # ---------------------------------------------------------------
        # SEARCH (no track data, no gate visible)
        # ---------------------------------------------------------------
        else:
            self._navigate_search(drone_yaw)

    def _navigate_vision_only(self, yaw, vision):
        """Fly toward detected gate using vision (when no track data)."""
        px_err = vision.get('pixel_error', (0, 0))
        distance = vision.get('distance', 10)

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Speed based on distance
        if distance is not None and distance < 5:
            speed = PRECISION_SPEED
        elif distance is not None and distance < 10:
            speed = APPROACH_SPEED
        else:
            speed = CRUISE_SPEED if distance is None else MAX_SPEED

        # Body-frame corrections from pixel error
        lateral = clamp(px_err[0] * 0.005, -2.0, 2.0)
        vertical = clamp(px_err[1] * 0.003, -1.5, 1.5)

        # Convert to NED
        vx_ned = cos_y * speed - sin_y * lateral
        vy_ned = sin_y * speed + cos_y * lateral
        vz_ned = vertical

        yaw_rate = clamp(px_err[0] * 0.004, -1.5, 1.5)

        self._send_velocity_ned(vx_ned, vy_ned, vz_ned, yaw_rate)

    def _navigate_search(self, yaw):
        """No gate visible — fly forward and search."""
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        vx = cos_y * SEARCH_SPEED
        vy = sin_y * SEARCH_SPEED

        time_since = time.time() - self.last_gate_seen_time if self.last_gate_seen_time > 0 else 999

        if time_since > NO_GATE_TIMEOUT:
            yr = SEARCH_YAW_RATE * self.search_direction
            if int(time_since) % 6 < 3:
                self.search_direction = 1.0
            else:
                self.search_direction = -1.0
        else:
            yr = 0.2 * self.search_direction

        self._send_velocity_ned(vx, vy, 0.0, yr)

    def _handle_finished(self):
        """Race complete — hover."""
        self._send_velocity_ned(0.0, 0.0, 0.0, 0.0)

    # ===================================================================
    # COMMAND SENDERS
    # ===================================================================

    def _send_idle(self):
        """Send zero-throttle commands (for arming)."""
        now_ms = int(time.time() * 1000)

        # Zero-thrust attitude command
        self.sim_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            RATES_ATTITUDE_MASK,
            [1, 0, 0, 0],
            0.0, 0.0, 0.0,   # zero rates
            0.0               # zero thrust!
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
            0.0, 0.0, 0.0,                          # position (ignored)
            float(vx), float(vy), float(vz),         # velocity
            0.0, 0.0, 0.0,                            # acceleration (ignored)
            0.0,                                       # yaw (ignored)
            float(yaw_rate)                            # yaw rate
        )

    def _send_attitude(self, roll_rate, pitch_rate, yaw_rate, thrust):
        """Send attitude rate + thrust command."""
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            RATES_ATTITUDE_MASK,
            [1, 0, 0, 0],
            float(roll_rate),
            float(pitch_rate),
            float(yaw_rate),
            float(thrust)
        )

    # ===================================================================
    # HELPERS
    # ===================================================================

    def _load_track_data(self):
        track = self.data.get('track', {})
        self.gates = track.get('gates', [])
        self.num_gates = track.get('num_gates', 0)
        if self.num_gates > 0:
            print(f"[CTRL] Loaded {self.num_gates} gates for navigation.", flush=True)

    def _change_state(self, new_state):
        old = self.state
        self.state = new_state
        self.state_start_time = time.time()
        print(f"[CTRL] State: {old} → {new_state}", flush=True)

    def _print_status(self):
        att = self.data.get('attitude', {})
        pos = self.data.get('local_position', {})
        race = self.data.get('race_status', {})

        pitch_deg = math.degrees(att.get('pitch', 0))
        roll_deg = math.degrees(att.get('roll', 0))
        yaw_deg = math.degrees(att.get('yaw', 0))

        x, y, z = pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)
        vx, vy, vz = pos.get('vx', 0), pos.get('vy', 0), pos.get('vz', 0)
        speed = math.sqrt(vx**2 + vy**2 + vz**2)

        active_gate = race.get('active_gate_index', self.target_gate_index)
        armed = self.data.get('armed', False)

        vision = self.data.get('vision_detection', {})
        vis_fresh = (time.time() - vision.get('timestamp', 0)) < 0.5
        gate_vis = vision.get('detected', False) and vis_fresh

        vis_str = "NO"
        if gate_vis:
            d = vision.get('distance', 0)
            px = vision.get('pixel_error', (0, 0))
            vis_str = f"YES(d={d:.1f}m px=({px[0]:.0f},{px[1]:.0f}))"

        dist_str = ""
        if self.num_gates > 0 and self.target_gate_index < self.num_gates:
            gate_pos = np.array(self.gates[self.target_gate_index]['position'])
            drone_pos = np.array([x, y, z])
            dist = np.linalg.norm(gate_pos - drone_pos)
            dist_str = f" dist={dist:.1f}m"

        status = (f"[STATUS] {self.state} armed={armed} gate={active_gate}/{self.num_gates} "
                  f"pos=({x:.1f},{y:.1f},{z:.1f}) spd={speed:.1f}m/s "
                  f"pitch={pitch_deg:.0f}° yaw={yaw_deg:.0f}° "
                  f"vision={vis_str}{dist_str}")

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
