"""
Autonomous Flight Controller for AI Grand Prix
=================================================
State machine: WAIT (EKF) → ARMING → TAKEOFF (timer) → NAVIGATE → FINISHED

Uses NED velocity commands, track waypoints with Z-inverted gate altitudes,
optional vision blend, and no RC overrides (they hijack velocity control).
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

EKF_BOOT_DELAY_S = 4.0
ARM_SPAM_DURATION_S = 0.5
TAKEOFF_DURATION_S = 2.5
TAKEOFF_VZ = -2.5  # NED: negative = up

MAX_SPEED = 6.0
APPROACH_SPEED = 3.5
PRECISION_SPEED = 2.5
FAR_DISTANCE = 12.0
APPROACH_DISTANCE = 6.0
PRECISION_DISTANCE = 3.0
GATE_LOOKAHEAD = 1.5
AIM_ALTITUDE_NED = -1.0  # never aim below 1m above ground (NED)

VISION_BLEND_DIST = 8.0
VISION_GAIN = 0.005
MAX_VISION_WEIGHT = 0.3


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def wrap_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def quat_forward_vector(qw, qx, qy, qz):
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    return np.array([fx, fy, fz])


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.state = "WAIT"
        self.startup_time = time.time()
        self.takeoff_start = 0.0
        self.last_print_time = 0.0
        self.target_gate_idx = 0
        self.gates = []
        self.last_cmd = (0.0, 0.0, 0.0, 0.0)
        self._ekf_logged = False

        print("[CTRL] Controller initialized.", flush=True)

    def _force_guided_mode(self):
        try:
            self.sim_conn.mav.set_mode_send(
                self.sim_conn.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4,  # GUIDED
            )
        except Exception:
            pass

    def _load_track(self):
        track = self.data.get('track', {})
        gate_list = track.get('gates', [])
        self.gates = []
        for g in gate_list:
            raw = g['position']
            # Simulator track Z is altitude-up; ArduPilot NED uses Z-down.
            pos = np.array([raw[0], raw[1], -raw[2]])
            q = g['orientation']
            fwd = quat_forward_vector(q[0], q[1], q[2], q[3])
            self.gates.append({'pos': pos, 'fwd': fwd, 'raw_z': raw[2]})
        print(f"[CTRL] Track loaded: {len(self.gates)} gates (Z-inverted for NED).", flush=True)
        for i, g in enumerate(self.gates):
            p = g['pos']
            print(f"  Gate {i}: ned=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) raw_z={g['raw_z']:.1f}", flush=True)

    def update(self):
        now = time.time()
        elapsed = now - self.startup_time

        if self.state == "WAIT":
            race_started = self.data.get('race_status', {}).get('race_started', False)
            track_ready = self.data.get('track', {}).get('received', False)

            if elapsed < EKF_BOOT_DELAY_S:
                if not self._ekf_logged or int(elapsed * 2) != int((elapsed - 0.02) * 2):
                    print(f"[CTRL] Booting EKF... {elapsed:.1f}s", flush=True)
                    self._ekf_logged = True
            elif race_started or track_ready:
                self._force_guided_mode()
                if track_ready:
                    self._load_track()
                print("[CTRL] EKF ready. Arming...", flush=True)
                self.state = "ARMING"
                self.takeoff_start = now

        elif self.state == "ARMING":
            self.arm()
            if now - self.takeoff_start > ARM_SPAM_DURATION_S:
                print("[CTRL] Armed! Taking off!", flush=True)
                self.state = "TAKEOFF"
                self.takeoff_start = now

        elif self.state == "TAKEOFF":
            t_elapsed = now - self.takeoff_start
            if t_elapsed < TAKEOFF_DURATION_S:
                self._send_velocity_ned(0.0, 0.0, TAKEOFF_VZ, 0.0)
            else:
                print("[CTRL] Takeoff complete. Navigating to Gate 0!", flush=True)
                self.state = "NAVIGATE"

        elif self.state == "NAVIGATE":
            self._handle_navigate()

        elif self.state == "FINISHED":
            self._send_velocity_ned(0.0, 0.0, 0.0, 0.0)

        if now - self.last_print_time > 0.5:
            self._print_status()
            self.last_print_time = now

        time.sleep(0.02)

    def _handle_navigate(self):
        race = self.data.get('race_status', {})
        if race.get('race_finished', False):
            print("[CTRL] Race finished! Hovering.", flush=True)
            self.state = "FINISHED"
            self._send_velocity_ned(0.0, 0.0, 0.0, 0.0)
            return

        active_idx = race.get('active_gate_index', 0)
        if active_idx > self.target_gate_idx:
            print(f"[CTRL] Gate {self.target_gate_idx} cleared → {active_idx}", flush=True)
            self.target_gate_idx = active_idx

        if not self.gates or self.target_gate_idx >= len(self.gates):
            self._send_velocity_ned(0.0, 0.0, 0.0, 0.0)
            return

        gate = self.gates[self.target_gate_idx]
        gate_pos = gate['pos']
        gate_fwd = gate['fwd']

        pos = self.data.get('local_position', {})
        drone_pos = np.array([pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)])
        att = self.data.get('attitude', {})
        drone_yaw = att.get('yaw', 0.0)

        aim_point = gate_pos + gate_fwd * GATE_LOOKAHEAD
        aim_point[2] = min(aim_point[2], AIM_ALTITUDE_NED)

        to_aim = aim_point - drone_pos
        dist_to_gate = np.linalg.norm(gate_pos - drone_pos)
        direction = normalize(to_aim)

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

        vel_cmd = direction * speed

        vision = self.data.get('vision_detection') or {}
        if vision.get('detected', False) and dist_to_gate < VISION_BLEND_DIST:
            px_err = vision.get('pixel_error', (0, 0))
            weight = min(1.0 - (dist_to_gate / VISION_BLEND_DIST), MAX_VISION_WEIGHT)
            cos_yaw = math.cos(drone_yaw)
            sin_yaw = math.sin(drone_yaw)
            lat_corr = px_err[0] * VISION_GAIN * weight
            vert_corr = px_err[1] * VISION_GAIN * weight
            vel_cmd[0] += -sin_yaw * lat_corr
            vel_cmd[1] += cos_yaw * lat_corr
            vel_cmd[2] += vert_corr

        target_yaw = math.atan2(to_aim[1], to_aim[0])
        yaw_err = wrap_pi(target_yaw - drone_yaw)
        yaw_rate = clamp(yaw_err * 0.7, -1.0, 1.0)

        alignment_factor = max(0.1, math.cos(yaw_err))
        vel_cmd[0] *= alignment_factor
        vel_cmd[1] *= alignment_factor
        vel_cmd[2] *= alignment_factor

        speed_mag = np.linalg.norm(vel_cmd)
        if speed_mag > MAX_SPEED:
            vel_cmd = vel_cmd / speed_mag * MAX_SPEED

        self._send_velocity_ned(vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate)

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate):
        self.last_cmd = (vx, vy, vz, yaw_rate)
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_position_target_local_ned_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            VELOCITY_ONLY_MASK,
            0, 0, 0,
            float(vx), float(vy), float(vz),
            0, 0, 0,
            0,
            float(yaw_rate),
        )

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            31000, 0, 0, 0, 0, 0, 0, 0, 0,
        )

    def _print_status(self):
        pos = self.data.get('local_position', {})
        att = self.data.get('attitude', {})
        x, y, z = pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)
        vx, vy, vz = pos.get('vx', 0), pos.get('vy', 0), pos.get('vz', 0)
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        yaw_deg = math.degrees(att.get('yaw', 0))
        armed = self.data.get('armed', False)
        cmd = self.last_cmd

        line = (
            f"[STATUS] {self.state} armed={armed} "
            f"pos=({x:.1f},{y:.1f},{z:.1f}) speed={speed:.1f}m/s yaw={yaw_deg:.0f}° "
            f"cmd=({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f},{cmd[3]:.2f})"
        )

        if self.gates and self.target_gate_idx < len(self.gates):
            gp = self.gates[self.target_gate_idx]['pos']
            dist = np.linalg.norm(gp - np.array([x, y, z]))
            line += f" gate={self.target_gate_idx}/{len(self.gates)} dist={dist:.1f}m"

        print(line, flush=True)
