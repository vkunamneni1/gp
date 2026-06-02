"""
Autonomous Flight Controller for AI Grand Prix
=================================================
State-machine-based controller that navigates through race gates using:
  1. Waypoint navigation (track data gate positions + local NED position)
  2. Vision refinement (gate detector for precision approach)
  3. Velocity-based control via SET_POSITION_TARGET_LOCAL_NED

Control hierarchy:
  IDLE → WAIT_FOR_DATA → TAKEOFF → NAVIGATE → FINISHED

Focus: COURSE COMPLETION. Safe, reliable, gate-by-gate navigation.
"""

import time
import math
import numpy as np
from pymavlink import mavutil

# --------------------------------------------------------------------------------------
# RESET COMMAND
MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# Flight States
# --------------------------------------------------------------------------------------
STATE_IDLE            = "IDLE"
STATE_WAIT_FOR_DATA   = "WAIT_FOR_DATA"
STATE_TAKEOFF         = "TAKEOFF"
STATE_NAVIGATE        = "NAVIGATE"
STATE_RECOVER         = "RECOVER"
STATE_FINISHED        = "FINISHED"

# --------------------------------------------------------------------------------------
# Tuning Parameters
# --------------------------------------------------------------------------------------
CONTROL_HZ = 50           # 50 Hz control loop

# Speed profile
MAX_SPEED       = 6.0     # m/s — cruise speed
APPROACH_SPEED  = 3.5     # m/s — when getting close to gate
PRECISION_SPEED = 2.5     # m/s — very close to gate, precision threading
MIN_SPEED       = 1.5     # m/s — minimum forward speed

# Distance thresholds
FAR_DISTANCE       = 12.0   # > this = cruise speed
APPROACH_DISTANCE  = 6.0    # < this = approach speed
PRECISION_DISTANCE = 3.0    # < this = precision speed
GATE_PASSED_DIST   = 1.5    # consider gate passed if within this distance

# Takeoff
TAKEOFF_ALT   = -2.0      # NED: negative = up. 2m above ground.
TAKEOFF_SPEED = 2.0        # m/s upward
ALT_TOLERANCE = 0.5        # meters

# Gate look-ahead: how far past the gate center to aim (to ensure clean pass-through)
GATE_LOOKAHEAD = 2.5       # meters past gate center along gate normal

# Vision blending
VISION_BLEND_DISTANCE = 8.0    # start blending vision corrections within this distance
VISION_WEIGHT_MAX     = 0.3    # max weight for vision corrections
VISION_PIXEL_GAIN     = 0.003  # how much pixel error translates to velocity correction

# Collision recovery
RECOVERY_DURATION = 1.0   # seconds to back off after collision
RECOVERY_SPEED    = -1.5  # back up slowly


def quat_forward_vector(qw, qx, qy, qz):
    """
    Get the forward direction (through the gate) from gate orientation quaternion.
    Rotates unit-X by the quaternion.
    """
    fx = 1.0 - 2.0*(qy*qy + qz*qz)
    fy = 2.0*(qx*qy + qw*qz)
    fz = 2.0*(qx*qz - qw*qy)
    return np.array([fx, fy, fz])


def normalize(v):
    """Normalize a vector, return zero vector if magnitude is too small."""
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


# --------------------------------------------------------------------------------------
# Velocity-based position target mask: use only velocity fields + yaw_rate
# --------------------------------------------------------------------------------------
VELOCITY_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    # Note: NOT ignoring YAW_RATE — we use it for heading control
)


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms

        # State machine
        self.state = STATE_IDLE
        self.state_start_time = time.time()

        # Navigation state
        self.target_gate_index = 0
        self.gates = []
        self.num_gates = 0

        # Recovery
        self.recovery_start = 0

        # Timing
        self.last_print_time = 0
        self.loop_count = 0

        # Startup delay — give time for telemetry to arrive
        self.startup_time = time.time()

        # Arming control
        self.armed_once = False

        print("[CTRL] Controller initialized. Waiting for data...", flush=True)

    def update(self):
        """Main control loop tick — called at CONTROL_HZ."""
        self.loop_count += 1
        now = time.time()

        # Send RC overrides to prevent RC-loss failsafe.
        # CRITICAL: Throttle (ch3) MUST be 1000, not 1500.
        # 1500 = mid-throttle = "hover command" that fights our velocity commands.
        # 1000 = no throttle input = velocity commands have full authority.
        self.sim_conn.mav.rc_channels_override_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            1500, 1500, 1000, 1500, 0, 0, 0, 0
        )

        # Keep spamming GUIDED mode every tick to make sure it sticks
        if self.state in (STATE_WAIT_FOR_DATA, STATE_TAKEOFF, STATE_NAVIGATE):
            try:
                self.sim_conn.mav.set_mode_send(
                    self.sim_conn.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    4  # GUIDED
                )
            except Exception:
                pass

        # ---------------------------------------------------------------
        # STATE MACHINE
        # ---------------------------------------------------------------
        if self.state == STATE_IDLE:
            self._handle_idle()

        elif self.state == STATE_WAIT_FOR_DATA:
            self._handle_wait_for_data()

        elif self.state == STATE_TAKEOFF:
            self._handle_takeoff()

        elif self.state == STATE_NAVIGATE:
            self._handle_navigate()

        elif self.state == STATE_RECOVER:
            self._handle_recover()

        elif self.state == STATE_FINISHED:
            self._handle_finished()

        # Periodic status print
        if now - self.last_print_time > 2.0:
            self._print_status()
            self.last_print_time = now

        time.sleep(1.0 / CONTROL_HZ)

    # ===================================================================
    # STATE HANDLERS
    # ===================================================================

    def _handle_idle(self):
        """Wait for EKF to stabilize after SIM_RESET, then move to data acquisition."""
        elapsed = time.time() - self.startup_time
        if elapsed > 3.0:
            # Force GUIDED mode before we do anything
            try:
                self.sim_conn.mav.set_mode_send(
                    self.sim_conn.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    4  # GUIDED
                )
            except Exception:
                pass
            self._change_state(STATE_WAIT_FOR_DATA)
        elif self.loop_count % 100 == 0:
            print(f"[CTRL] Waiting for EKF stabilization... {elapsed:.1f}s", flush=True)

    def _handle_wait_for_data(self):
        """Wait until we have position data and track data, then arm and take off."""
        has_pos = self.data.get('has_position', False)
        track_received = self.data.get('track', {}).get('received', False)
        race_started = self.data.get('race_status', {}).get('race_started', False)

        elapsed = time.time() - self.state_start_time

        # Print waiting status periodically
        if self.loop_count % 150 == 0:
            print(f"[CTRL] Waiting... pos={has_pos} track={track_received} race={race_started} ({elapsed:.0f}s)", flush=True)

        if has_pos and (track_received or elapsed > 10.0):
            if track_received:
                self._load_track_data()
            else:
                print("[CTRL] No track data received — proceeding with vision-only navigation", flush=True)

            # Arm the drone (spam a few times for UDP reliability)
            print("[CTRL] Arming drone...", flush=True)
            for _ in range(5):
                self.arm()
                time.sleep(0.05)

            self._change_state(STATE_TAKEOFF)

    def _handle_takeoff(self):
        """Ascend to flight altitude using time-based approach."""
        elapsed = time.time() - self.state_start_time

        # Keep spamming arm during first second in case packets were dropped
        if elapsed < 1.0:
            self.arm()

        # Time-based takeoff: go up for 3 seconds, then navigate
        if elapsed < 3.0:
            self._send_velocity_ned(0.0, 0.0, -TAKEOFF_SPEED, 0.0)
        else:
            pos = self.data.get('local_position', {})
            current_z = pos.get('z', 0.0)
            print(f"[CTRL] Takeoff complete at z={current_z:.2f}m. Starting navigation!", flush=True)
            self._change_state(STATE_NAVIGATE)

    def _handle_navigate(self):
        """
        Main navigation logic — fly through gates sequentially.
        Uses waypoint navigation with optional vision refinement.
        """
        # Check race status
        race = self.data.get('race_status', {})
        if race.get('race_finished', False):
            self._change_state(STATE_FINISHED)
            return

        # Update target gate from race status
        race_gate_idx = race.get('active_gate_index', 0)
        if race_gate_idx > self.target_gate_index:
            print(f"[CTRL] Gate {self.target_gate_index} cleared! Now targeting gate {race_gate_idx}", flush=True)
            self.target_gate_index = race_gate_idx

        # Check for collisions — enter recovery if bad
        collision = self.data.get('collision', {})
        if collision.get('active', False):
            col_time = collision.get('timestamp', 0)
            if time.time() - col_time < 0.5:  # recent collision
                threat = collision.get('threat_level', 0)
                if threat >= 2:
                    print(f"[CTRL] Heavy collision detected! Entering recovery...", flush=True)
                    self._change_state(STATE_RECOVER)
                    return
                # Clear collision flag for light impacts
                self.data['collision']['active'] = False

        # Get drone position
        pos = self.data.get('local_position', {})
        drone_pos = np.array([pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)])

        # Get drone yaw
        att = self.data.get('attitude', {})
        drone_yaw = att.get('yaw', 0.0)

        # Determine target
        if self.num_gates > 0 and self.target_gate_index < self.num_gates:
            gate = self.gates[self.target_gate_index]
            gate_pos = np.array(gate['position'])
            gate_ori = gate['orientation']  # (w, x, y, z)

            # Compute gate forward direction
            gate_forward = quat_forward_vector(*gate_ori)
            gate_forward_2d = normalize(np.array([gate_forward[0], gate_forward[1], 0.0]))

            # Aim point: slightly past the gate center along its forward direction
            # This ensures we fly THROUGH the gate, not stop at it
            aim_point = gate_pos + gate_forward_2d * GATE_LOOKAHEAD

            # If there's a next gate, blend the aim direction toward it for smooth turns
            if self.target_gate_index + 1 < self.num_gates:
                next_gate_pos = np.array(self.gates[self.target_gate_index + 1]['position'])
                next_dir = normalize(next_gate_pos - gate_pos)
                dist_to_gate = np.linalg.norm(gate_pos - drone_pos)
                if dist_to_gate < APPROACH_DISTANCE:
                    blend = 1.0 - (dist_to_gate / APPROACH_DISTANCE)
                    blend = blend * 0.3  # subtle blending
                    aim_point = aim_point + next_dir * blend * 3.0

            # Vector from drone to aim point
            to_target = aim_point - drone_pos
            dist_to_gate = np.linalg.norm(gate_pos - drone_pos)
            direction = normalize(to_target)

            # Speed profile based on distance to gate
            if dist_to_gate > FAR_DISTANCE:
                target_speed = MAX_SPEED
            elif dist_to_gate > APPROACH_DISTANCE:
                t = (dist_to_gate - APPROACH_DISTANCE) / (FAR_DISTANCE - APPROACH_DISTANCE)
                target_speed = APPROACH_SPEED + t * (MAX_SPEED - APPROACH_SPEED)
            elif dist_to_gate > PRECISION_DISTANCE:
                t = (dist_to_gate - PRECISION_DISTANCE) / (APPROACH_DISTANCE - PRECISION_DISTANCE)
                target_speed = PRECISION_SPEED + t * (APPROACH_SPEED - PRECISION_SPEED)
            else:
                target_speed = PRECISION_SPEED

            # Compute velocity command
            vel_cmd = direction * target_speed

            # Vision refinement — apply corrections from gate detector
            vision = self.data.get('vision_detection', None)
            if vision is not None and vision.get('detected', False):
                vis_dist = vision.get('distance', 999)
                if vis_dist < VISION_BLEND_DISTANCE and dist_to_gate < VISION_BLEND_DISTANCE:
                    pixel_err = vision.get('pixel_error', (0, 0))

                    # Weight increases as we get closer
                    t = 1.0 - (dist_to_gate / VISION_BLEND_DISTANCE)
                    weight = t * VISION_WEIGHT_MAX

                    cos_yaw = math.cos(drone_yaw)
                    sin_yaw = math.sin(drone_yaw)

                    # Lateral correction in NED frame
                    lat_correction = pixel_err[0] * VISION_PIXEL_GAIN * weight
                    # Vertical correction in NED
                    vert_correction = pixel_err[1] * VISION_PIXEL_GAIN * weight

                    vel_cmd[0] += -sin_yaw * lat_correction
                    vel_cmd[1] += cos_yaw * lat_correction
                    vel_cmd[2] += vert_correction

            # Yaw rate: point toward the aim point
            desired_yaw = math.atan2(to_target[1], to_target[0])
            yaw_error = desired_yaw - drone_yaw
            # Wrap to [-pi, pi]
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi
            yaw_rate = clamp(yaw_error * 2.0, -2.0, 2.0)  # P-controller for yaw

            # Clamp total velocity magnitude
            speed = np.linalg.norm(vel_cmd)
            if speed > MAX_SPEED:
                vel_cmd = vel_cmd / speed * MAX_SPEED

            self._send_velocity_ned(vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate)

        elif self.num_gates == 0:
            # No track data — use vision-only navigation
            self._navigate_vision_only(drone_yaw)

        else:
            # All gates completed
            print("[CTRL] All gates completed! Hovering...", flush=True)
            self._send_velocity_ned(0.0, 0.0, 0.0, 0.0)
            self._change_state(STATE_FINISHED)

    def _navigate_vision_only(self, drone_yaw):
        """
        Fallback navigation using only vision detection.
        Fly toward detected gate, or search if no gate visible.
        """
        vision = self.data.get('vision_detection', None)

        if vision is not None and vision.get('detected', False):
            tvec = vision.get('tvec_body_ned', None)
            if tvec is not None:
                direction = normalize(tvec)
                distance = np.linalg.norm(tvec)

                if distance > FAR_DISTANCE:
                    speed = MAX_SPEED
                elif distance > APPROACH_DISTANCE:
                    speed = APPROACH_SPEED
                else:
                    speed = PRECISION_SPEED

                vel = direction * speed

                desired_yaw = math.atan2(tvec[1], tvec[0]) + drone_yaw
                yaw_error = desired_yaw - drone_yaw
                while yaw_error > math.pi:
                    yaw_error -= 2 * math.pi
                while yaw_error < -math.pi:
                    yaw_error += 2 * math.pi
                yaw_rate = clamp(yaw_error * 2.0, -2.0, 2.0)

                cos_y = math.cos(drone_yaw)
                sin_y = math.sin(drone_yaw)
                vx_ned = cos_y * vel[0] - sin_y * vel[1]
                vy_ned = sin_y * vel[0] + cos_y * vel[1]
                vz_ned = vel[2]

                self._send_velocity_ned(vx_ned, vy_ned, vz_ned, yaw_rate)
                return

        # No gate visible — search pattern: fly forward slowly, slight yaw to scan
        search_speed = 2.0
        cos_y = math.cos(drone_yaw)
        sin_y = math.sin(drone_yaw)
        self._send_velocity_ned(
            cos_y * search_speed,
            sin_y * search_speed,
            0.0,
            0.3  # slow yaw to scan
        )

    def _handle_recover(self):
        """Back up briefly after a collision, then resume navigation."""
        elapsed = time.time() - self.state_start_time

        if elapsed > RECOVERY_DURATION:
            self.data['collision']['active'] = False
            print("[CTRL] Recovery complete, resuming navigation.", flush=True)
            self._change_state(STATE_NAVIGATE)
            return

        att = self.data.get('attitude', {})
        yaw = att.get('yaw', 0.0)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Reverse along current heading
        self._send_velocity_ned(
            cos_y * RECOVERY_SPEED,
            sin_y * RECOVERY_SPEED,
            -0.5,   # go up a bit
            0.0
        )

    def _handle_finished(self):
        """Race complete — hover in place."""
        self._send_velocity_ned(0.0, 0.0, 0.0, 0.0)

    # ===================================================================
    # COMMAND SENDERS
    # ===================================================================

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate):
        """
        Send velocity command in NED frame.
        vx: North velocity (m/s)
        vy: East velocity (m/s)
        vz: Down velocity (m/s) — negative = up
        yaw_rate: yaw rate (rad/s)
        """
        now_ms = int(time.time() * 1000)

        self.sim_conn.mav.set_position_target_local_ned_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            VELOCITY_ONLY_MASK,
            0.0, 0.0, 0.0,          # position (ignored)
            float(vx), float(vy), float(vz),  # velocity
            0.0, 0.0, 0.0,          # acceleration (ignored)
            0.0,                     # yaw (ignored)
            float(yaw_rate)          # yaw rate
        )

    # ===================================================================
    # HELPERS
    # ===================================================================

    def _load_track_data(self):
        """Load gate positions from track data.
        
        NOTE: The track data from mavlink_rx.py already comes in NED coordinates
        (position_ned_x, position_ned_y, position_ned_z). No coordinate inversion needed.
        """
        track = self.data.get('track', {})
        self.gates = track.get('gates', [])
        self.num_gates = track.get('num_gates', 0)
        if self.num_gates > 0:
            print(f"[CTRL] Loaded {self.num_gates} gates for navigation.", flush=True)
            for i, g in enumerate(self.gates):
                p = g['position']
                print(f"  Gate {i}: pos=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})", flush=True)

    def _change_state(self, new_state):
        """Transition to a new state."""
        old = self.state
        self.state = new_state
        self.state_start_time = time.time()
        print(f"[CTRL] State: {old} → {new_state}", flush=True)

    def _print_status(self):
        """Print periodic status update."""
        pos = self.data.get('local_position', {})
        att = self.data.get('attitude', {})
        race = self.data.get('race_status', {})

        x, y, z = pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)
        vx, vy, vz = pos.get('vx', 0), pos.get('vy', 0), pos.get('vz', 0)
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        yaw_deg = math.degrees(att.get('yaw', 0))
        armed = self.data.get('armed', False)

        gate_idx = race.get('active_gate_index', self.target_gate_index)

        status = (f"[STATUS] state={self.state} armed={armed} gate={gate_idx}/{self.num_gates} "
                  f"pos=({x:.1f},{y:.1f},{z:.1f}) speed={speed:.1f}m/s yaw={yaw_deg:.0f}°")

        if self.num_gates > 0 and self.target_gate_index < self.num_gates:
            gate_pos = np.array(self.gates[self.target_gate_index]['position'])
            drone_pos = np.array([x, y, z])
            dist = np.linalg.norm(gate_pos - drone_pos)
            status += f" dist_to_gate={dist:.1f}m"

        vision = self.data.get('vision_detection', None)
        if vision and vision.get('detected', False):
            status += f" vision=YES(d={vision.get('distance', 0):.1f}m)"
        else:
            status += " vision=NO"

        print(status, flush=True)

    # ===================================================================
    # ARM / RESET
    # ===================================================================

    def arm(self):
        """Arm the drone."""
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # arm
            0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,  # confirmation
            0, 0, 0, 0, 0, 0, 0
        )
