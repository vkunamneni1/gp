"""
Autonomous Flight Controller for AI Grand Prix — ULTIMATE EDITION
===================================================================
A highly optimized, race-winning AI pilot.
Features:
1. Dynamic Speed Control: Slows down for sharp turns, accelerates on straights.
2. Smooth Trajectory: Blends current gate direction with next gate direction for perfect racing lines.
3. Vision PD Controller: Uses Proportional-Derivative control on camera pixel error to perfectly thread the needle.
4. Clean Startup: Waits for race start to avoid DQ, arms cleanly without triggering ACRO mode.
"""

import time
import math
import numpy as np
from pymavlink import mavutil

MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------
STATE_WAIT_DATA      = "WAIT_DATA"
STATE_WAIT_RACE      = "WAIT_RACE"
STATE_TAKEOFF        = "TAKEOFF"
STATE_NAVIGATE       = "NAVIGATE"
STATE_FINISHED       = "FINISHED"

# --------------------------------------------------------------------------------------
# Tuning Parameters
# --------------------------------------------------------------------------------------
CONTROL_HZ = 50

# Dynamic Speed Profile
MAX_SPEED       = 9.0    # Blast down straights
MIN_SPEED       = 3.5    # Minimum cornering speed
TAKEOFF_ALT     = -2.0   # 2m up
TAKEOFF_SPEED   = 2.0    # Fast liftoff

# Vision PD Controller
VISION_BLEND_DISTANCE = 12.0
KP_VISION = 0.005
KD_VISION = 0.002

VELOCITY_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6: return np.zeros_like(v)
    return v / n

def quat_forward_vector(qw, qx, qy, qz):
    fx = 1.0 - 2.0*(qy*qy + qz*qz)
    fy = 2.0*(qx*qy + qw*qz)
    fz = 2.0*(qx*qz - qw*qy)
    return np.array([fx, fy, fz])

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        
        self.state = STATE_WAIT_DATA
        self.state_start_time = time.time()
        
        self.target_gate_index = 0
        self.gates = []
        self.num_gates = 0
        
        self.last_pixel_error = (0, 0)
        self.last_print_time = 0
        self.loop_count = 0

        print("[CTRL] ULTIMATE Pilot Initialized. Waiting for data...", flush=True)

    def update(self):
        self.loop_count += 1
        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == STATE_WAIT_DATA:
            self._handle_wait_data(elapsed)
        elif self.state == STATE_WAIT_RACE:
            self._handle_wait_race()
        elif self.state == STATE_TAKEOFF:
            self._handle_takeoff(elapsed)
        elif self.state == STATE_NAVIGATE:
            self._handle_navigate()
        elif self.state == STATE_FINISHED:
            self._send_velocity_ned(0,0,0,0)

        if now - self.last_print_time > 2.0:
            self._print_status()
            self.last_print_time = now

        time.sleep(1.0 / CONTROL_HZ)

    def _handle_wait_data(self, elapsed):
        has_pos = self.data.get('has_position', False)
        track_received = self.data.get('track', {}).get('received', False)
        
        if has_pos and track_received:
            self._load_track_data()
            self._change_state(STATE_WAIT_RACE)
        elif elapsed > 15.0 and has_pos:
            self._change_state(STATE_WAIT_RACE)

    def _handle_wait_race(self):
        race = self.data.get('race_status', {})
        if race.get('race_started', False):
            print("[CTRL] RACE STARTED! Taking off...", flush=True)
            self._change_state(STATE_TAKEOFF)
        else:
            # Send zero velocity to hold position on ground, keeps FC happy
            self._send_velocity_ned(0.0, 0.0, 0.0, 0.0)

    def _handle_takeoff(self, elapsed):
        pos = self.data.get('local_position', {})
        z = pos.get('z', 0.0)
        
        if z > TAKEOFF_ALT + 0.5:
            self._send_velocity_ned(0.0, 0.0, -TAKEOFF_SPEED, 0.0)
        else:
            self._change_state(STATE_NAVIGATE)

    def _handle_navigate(self):
        race = self.data.get('race_status', {})
        if race.get('race_finished', False):
            self._change_state(STATE_FINISHED)
            return

        race_gate_idx = race.get('active_gate_index', 0)
        if race_gate_idx > self.target_gate_index:
            self.target_gate_index = race_gate_idx

        pos = self.data.get('local_position', {})
        drone_pos = np.array([pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)])
        drone_yaw = self.data.get('attitude', {}).get('yaw', 0.0)

        if self.num_gates > 0 and self.target_gate_index < self.num_gates:
            gate = self.gates[self.target_gate_index]
            gate_pos = np.array(gate['position'])
            gate_dir = normalize(quat_forward_vector(*gate['orientation']))
            
            dist_to_gate = np.linalg.norm(gate_pos - drone_pos)
            
            # --- 1. Dynamic Racing Line ---
            # Aim slightly past the gate to ensure we fly through it cleanly
            aim_point = gate_pos + gate_dir * 1.5
            
            # Blend next gate for smooth cornering
            target_speed = MAX_SPEED
            if self.target_gate_index + 1 < self.num_gates:
                next_gate_pos = np.array(self.gates[self.target_gate_index + 1]['position'])
                dir_to_next = normalize(next_gate_pos - gate_pos)
                
                # Calculate turn sharpness
                dot_prod = clamp(np.dot(gate_dir, dir_to_next), -1.0, 1.0)
                turn_angle = math.acos(dot_prod)
                
                # Slow down proportional to turn sharpness
                turn_factor = 1.0 - (turn_angle / math.pi)
                target_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * turn_factor
                
                # Start cutting the corner slightly if we're close
                if dist_to_gate < 5.0:
                    blend = (1.0 - dist_to_gate / 5.0) * 0.4
                    aim_point = aim_point + dir_to_next * blend * 4.0
            
            if dist_to_gate < 3.0:
                target_speed = MIN_SPEED
                
            dir_to_aim = normalize(aim_point - drone_pos)
            vel_cmd = dir_to_aim * target_speed
            
            # --- 2. Vision PD Refinement ---
            vision = self.data.get('vision_detection', {})
            vis_fresh = (time.time() - vision.get('timestamp', 0)) < 0.2
            
            if vis_fresh and vision.get('detected', False) and dist_to_gate < VISION_BLEND_DISTANCE:
                px_err = vision.get('pixel_error', (0, 0))
                
                # PD Control
                p_err = np.array(px_err)
                d_err = p_err - np.array(self.last_pixel_error)
                self.last_pixel_error = px_err
                
                correction = p_err * KP_VISION + d_err * KD_VISION
                
                # Weight increases aggressively as we get closer to the gate
                weight = 1.0 - (dist_to_gate / VISION_BLEND_DISTANCE)
                weight = weight ** 2  # Exponential weight for precision finish
                
                cos_yaw, sin_yaw = math.cos(drone_yaw), math.sin(drone_yaw)
                
                # Horizontal error -> Lateral velocity (Y in body frame)
                lat_corr = correction[0] * weight
                vel_cmd[0] += -sin_yaw * lat_corr
                vel_cmd[1] += cos_yaw * lat_corr
                
                # Vertical error -> Vertical velocity (Z in NED)
                vel_cmd[2] += correction[1] * weight

            # --- 3. Heading Control ---
            desired_yaw = math.atan2(dir_to_aim[1], dir_to_aim[0])
            yaw_err = desired_yaw - drone_yaw
            while yaw_err > math.pi: yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi
            yaw_rate = clamp(yaw_err * 2.5, -2.5, 2.5)

            # Enforce max speed
            spd = np.linalg.norm(vel_cmd)
            if spd > MAX_SPEED:
                vel_cmd = (vel_cmd / spd) * MAX_SPEED

            self._send_velocity_ned(vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate)
            
        else:
            self._navigate_vision_only(drone_yaw)

    def _navigate_vision_only(self, yaw):
        self._send_velocity_ned(math.cos(yaw)*2, math.sin(yaw)*2, 0, 0.2)

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate):
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_position_target_local_ned_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            VELOCITY_ONLY_MASK,
            0, 0, 0,
            float(vx), float(vy), float(vz),
            0, 0, 0, 0, float(yaw_rate)
        )

    def _load_track_data(self):
        track = self.data.get('track', {})
        self.gates = track.get('gates', [])
        self.num_gates = track.get('num_gates', 0)
        print(f"[CTRL] Track loaded: {self.num_gates} gates.", flush=True)

    def _change_state(self, new_state):
        print(f"[CTRL] {self.state} -> {new_state}", flush=True)
        self.state = new_state
        self.state_start_time = time.time()

    def _print_status(self):
        race = self.data.get('race_status', {})
        pos = self.data.get('local_position', {})
        gate_idx = race.get('active_gate_index', self.target_gate_index)
        speed = math.sqrt(pos.get('vx',0)**2 + pos.get('vy',0)**2 + pos.get('vz',0)**2)
        print(f"[STATUS] {self.state} | Gate: {gate_idx}/{self.num_gates} | Speed: {speed:.1f}m/s", flush=True)

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )
