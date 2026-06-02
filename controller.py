"""
Autonomous Flight Controller for AI Grand Prix
=================================================
Simplified, proven controller that matches the version the user confirmed working.
State machine: WAIT → ARMING → TAKEOFF → NAVIGATE
"""

import time
import math
import numpy as np
from pymavlink import mavutil

VELOCITY_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

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
        self.state = "WAIT"
        self.startup_time = time.time()
        self.takeoff_start = 0
        self.target_gate_idx = 0
        self.gates = []
        self.num_gates = 0
        self.last_cmd = (0, 0, 0, 0)

        print("[CTRL] Controller initialized.", flush=True)

        # Set GUIDED mode once at init (before SIM_RESET, but it may persist)
        try:
            self.sim_conn.mav.set_mode_send(
                self.sim_conn.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4  # GUIDED
            )
        except:
            pass

    def _load_track(self):
        track = self.data.get('track', {})
        raw_gates = track.get('gates', [])
        self.gates = []
        for g in raw_gates:
            pos = g['position']
            self.gates.append({
                'position': np.array([pos[0], pos[1], pos[2]]),  # Use as-is from simulator
                'orientation': g['orientation'],
            })
        self.num_gates = len(self.gates)
        print(f"[CTRL] Track loaded: {self.num_gates} gates.", flush=True)
        for i, g in enumerate(self.gates):
            p = g['position']
            print(f"  Gate {i}: pos=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})", flush=True)

    def update(self):
        now = time.time()
        elapsed = now - self.startup_time

        # RC overrides to prevent failsafe (throttle=1000 = no RC throttle input)
        self.sim_conn.mav.rc_channels_override_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            1500, 1500, 1000, 1500, 0, 0, 0, 0
        )

        if self.state == "WAIT":
            race_started = self.data.get('race_status', {}).get('race_started', False)
            track_ready = self.data.get('track', {}).get('received', False)
            if race_started or elapsed > 5.0:
                if track_ready:
                    self._load_track()
                print(f"[CTRL] Arming! (elapsed={elapsed:.1f}s)", flush=True)
                self.arm()
                self.state = "ARMING"
                self.takeoff_start = now

        elif self.state == "ARMING":
            self.arm()  # spam arm every tick
            if now - self.takeoff_start > 0.5:
                print("[CTRL] Armed! Taking off!", flush=True)
                self.state = "TAKEOFF"
                self.takeoff_start = now

        elif self.state == "TAKEOFF":
            t = now - self.takeoff_start
            if t < 3.0:
                self._send_velocity_ned(0, 0, -3.0, 0)
            else:
                print("[CTRL] Takeoff done. Navigating!", flush=True)
                self.state = "NAVIGATE"

        elif self.state == "NAVIGATE":
            self._handle_navigate()

        # Print status every 0.5 seconds
        if not hasattr(self, '_last_status') or now - self._last_status > 0.5:
            self._print_status()
            self._last_status = now

        time.sleep(0.02)

    def _handle_navigate(self):
        race = self.data.get('race_status', {})
        if race.get('race_finished', False):
            self._send_velocity_ned(0, 0, 0, 0)
            return

        # Update target gate from race
        race_gate_idx = race.get('active_gate_index', 0)
        if race_gate_idx > self.target_gate_idx:
            print(f"[CTRL] Gate {self.target_gate_idx} cleared! → {race_gate_idx}", flush=True)
            self.target_gate_idx = race_gate_idx

        if self.num_gates == 0 or self.target_gate_idx >= self.num_gates:
            self._send_velocity_ned(0, 0, 0, 0)
            return

        # Get drone state
        pos = self.data.get('local_position', {})
        drone_pos = np.array([pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)])
        att = self.data.get('attitude', {})
        drone_yaw = att.get('yaw', 0.0)

        # Get gate
        gate = self.gates[self.target_gate_idx]
        gate_pos = gate['position']

        # Fly toward gate at 3 m/s
        to_gate = gate_pos - drone_pos
        dist = np.linalg.norm(to_gate)
        direction = normalize(to_gate)

        speed = 3.0
        vel_cmd = direction * speed

        # Yaw toward gate
        desired_yaw = math.atan2(to_gate[1], to_gate[0])
        yaw_err = desired_yaw - drone_yaw
        while yaw_err > math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi
        yaw_rate = clamp(yaw_err * 1.0, -1.0, 1.0)

        self._send_velocity_ned(vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate)

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate):
        self.last_cmd = (vx, vy, vz, yaw_rate)
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_position_target_local_ned_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, VELOCITY_ONLY_MASK,
            0, 0, 0, float(vx), float(vy), float(vz), 0, 0, 0, 0, float(yaw_rate)
        )

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            31000, 0, 0, 0, 0, 0, 0, 0, 0
        )

    def _print_status(self):
        pos = self.data.get('local_position', {})
        att = self.data.get('attitude', {})
        x, y, z = pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)
        vx, vy, vz = pos.get('vx', 0), pos.get('vy', 0), pos.get('vz', 0)
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        yaw_deg = math.degrees(att.get('yaw', 0))
        armed = self.data.get('armed', False)
        cmd = self.last_cmd

        line = (f"[STATUS] {self.state} armed={armed} "
                f"pos=({x:.1f},{y:.1f},{z:.1f}) speed={speed:.1f}m/s yaw={yaw_deg:.0f}° "
                f"cmd=({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f},{cmd[3]:.2f})")

        if self.num_gates > 0 and self.target_gate_idx < self.num_gates:
            gp = self.gates[self.target_gate_idx]['position']
            dist = np.linalg.norm(gp - np.array([x, y, z]))
            line += f" gate={self.target_gate_idx} dist={dist:.1f}m"

        print(line, flush=True)
