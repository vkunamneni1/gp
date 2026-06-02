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

# --- Racing Tuning Parameters ---
TAKEOFF_ALT        = -2.5    # meters (NED is negative up)
TAKEOFF_SPEED      = 3.0     # m/s

MAX_SPEED          = 6.0     # m/s on long straights
APPROACH_SPEED     = 3.5     # m/s when nearing gate
PRECISION_SPEED    = 2.0     # m/s when very close

FAR_DISTANCE       = 12.0    # meters
APPROACH_DISTANCE  = 6.0     # meters
PRECISION_DISTANCE = 3.0     # meters

GATE_LOOKAHEAD     = 1.5     # meters past the gate center to aim for

VISION_BLEND_DIST  = 8.0     # distance to start trusting vision
VISION_GAIN        = 0.005   # pixel error to velocity multiplier
MAX_VISION_WEIGHT  = 0.4     # max ratio of vision velocity vs waypoint velocity

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
    # Rotates unit-X by quaternion to get forward vector in NED
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
        self.last_print_time = 0
        self.target_gate_idx = 0
        self.gates = []
        
        print("[CTRL] Ultimate Bypass Script Initialized.", flush=True)

        # Force GUIDED mode right at startup (like the GOOD script)
        try:
            self.sim_conn.mav.set_mode_send(
                self.sim_conn.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4 # GUIDED
            )
        except:
            pass

    def _load_track(self):
        track = self.data.get('track', {})
        gate_list = track.get('gates', [])
        self.gates = []
        for g in gate_list:
            # The simulator sends Z as UP, but our flight controller expects NED (Z is DOWN).
            # We MUST invert Z to prevent the drone from diving into the ground!
            pos = np.array([g['position'][0], g['position'][1], -g['position'][2]])
            q = g['orientation']
            fwd = quat_forward_vector(q[0], q[1], q[2], q[3])
            self.gates.append({'pos': pos, 'fwd': fwd})
        print(f"[CTRL] Track loaded: {len(self.gates)} gates (Z-inverted for NED).", flush=True)

    def update(self):
        now = time.time()
        elapsed = now - self.startup_time

        # Send RC Overrides continuously to prevent Throttle Failsafe / RC Loss timeout!
        # This keeps the flight controller awake so it allows us to arm later.
        self.sim_conn.mav.rc_channels_override_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            1500, 1500, 1000, 1500, 0, 0, 0, 0
        )

        if self.state == "WAIT":
            race_started = self.data.get('race_status', {}).get('race_started', False)
            
            # STRICT EKF BOOT DELAY:
            # We MUST wait at least 4.0 seconds after SIM_RESET before arming, 
            # even if the race has already started! Otherwise the EKF diverges into space.
            if elapsed < 4.0:
                if int(now * 10) % 20 == 0:
                    print(f"[CTRL] Booting EKF... {elapsed:.1f}s", flush=True)
            elif race_started:
                print(f"[CTRL] EKF Ready & Race ON. Forcing GUIDED mode and Arming!", flush=True)
                # Force GUIDED right before arming to guarantee we aren't stuck in ACRO
                try:
                    self.sim_conn.mav.set_mode_send(
                        self.sim_conn.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        4 # GUIDED
                    )
                except:
                    pass
                
                self._load_track()
                self.state = "ARMING"
                self.takeoff_start = now

        elif self.state == "ARMING":
            # Spam arm() for 0.5 seconds to guarantee the UDP packet isn't dropped!
            self.arm()
            if now - self.takeoff_start > 0.5:
                print(f"[CTRL] Armed! Taking off!", flush=True)
                self.state = "TAKEOFF"
                self.takeoff_start = now

        elif self.state == "TAKEOFF":
            # Gentler takeoff to prevent initial wobble
            t_elapsed = now - self.takeoff_start
            if t_elapsed < 2.5:
                self._send_velocity_ned(0, 0, -1.5, 0)
            else:
                print(f"[CTRL] Altitude reached! Flying forward to Gate 0!", flush=True)
                self.state = "NAVIGATE"

        elif self.state == "NAVIGATE":
            self._handle_navigate()

        if now - self.last_print_time > 1.0:
            self._print_status()
            self.last_print_time = now

        time.sleep(0.02)

    def _handle_navigate(self):
        race = self.data.get('race_status', {})
        if race.get('race_finished', False):
            print("[CTRL] RACE FINISHED! Hovering...", flush=True)
            self._send_velocity_ned(0, 0, 0, 0)
            return

        # Update target gate
        active_idx = race.get('active_gate_index', 0)
        if active_idx > self.target_gate_idx:
            print(f"[CTRL] Gate {self.target_gate_idx} cleared! Heading to {active_idx}", flush=True)
            self.target_gate_idx = active_idx

        if self.target_gate_idx >= len(self.gates):
            self._send_velocity_ned(0, 0, 0, 0)
            return

        gate = self.gates[self.target_gate_idx]
        gate_pos = gate['pos']
        gate_fwd = gate['fwd']

        pos = self.data.get('local_position', {})
        drone_pos = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
        
        att = self.data.get('attitude', {})
        drone_yaw = att.get('yaw', 0.0)

        # 1. Base Waypoint Navigation
        aim_point = gate_pos + gate_fwd * GATE_LOOKAHEAD
        
        # NEVER aim into the ground! If the gate is on the floor (Z=0), aim at least 1m above it.
        # In NED, negative is UP. So we want aim_point[2] to be <= -1.0
        aim_point[2] = min(aim_point[2], -1.0)
        
        to_aim = aim_point - drone_pos
        dist_to_gate = np.linalg.norm(gate_pos - drone_pos)
        direction = normalize(to_aim)

        # 2. Dynamic Speed Scaling
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

        # 3. Vision Correction Mixing
        vision = self.data.get('vision_detection', {})
        if vision.get('detected', False) and dist_to_gate < VISION_BLEND_DIST:
            px_err = vision.get('pixel_error', (0, 0))
            weight = 1.0 - (dist_to_gate / VISION_BLEND_DIST)
            weight = min(weight, MAX_VISION_WEIGHT)

            cos_yaw = math.cos(drone_yaw)
            sin_yaw = math.sin(drone_yaw)

            lat_corr = px_err[0] * VISION_GAIN * weight
            vert_corr = px_err[1] * VISION_GAIN * weight

            vel_cmd[0] += -sin_yaw * lat_corr
            vel_cmd[1] += cos_yaw * lat_corr
            vel_cmd[2] += vert_corr

        # 4. Heading Alignment (Use stable to_aim, NOT the vision-perturbed vel_cmd!)
        target_yaw = math.atan2(to_aim[1], to_aim[0])
        yaw_err = wrap_pi(target_yaw - drone_yaw)
        # Use a gentler P-gain to prevent yaw wobble
        yaw_rate = clamp(yaw_err * 0.7, -1.0, 1.0)
        
        # 5. Speed Throttling (Don't fly 6 m/s backwards! Wait to turn first)
        # If yaw_err is large, cos() drops, slowing the drone down so it turns first.
        alignment_factor = max(0.1, math.cos(yaw_err))
        
        # Apply the throttling to our lateral/forward velocity
        vel_cmd[0] *= alignment_factor
        vel_cmd[1] *= alignment_factor
        vel_cmd[2] *= alignment_factor

        self._send_velocity_ned(vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate)

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

    def _print_status(self):
        pos = self.data.get('local_position', {})
        att = self.data.get('attitude', {})
        x, y, z = pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)
        yaw_deg = math.degrees(att.get('yaw', 0))
        
        status = f"[STATUS] state={self.state} pos=({x:.1f},{y:.1f},{z:.1f}) yaw={yaw_deg:.0f}deg"
        
        if self.state == "NAVIGATE" and self.target_gate_idx < len(self.gates):
            gate = self.gates[self.target_gate_idx]
            gp = gate['pos']
            status += f" | Gate={self.target_gate_idx} gpos=({gp[0]:.1f},{gp[1]:.1f},{gp[2]:.1f})"
            
        print(status, flush=True)

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            31000, 0, 0, 0, 0, 0, 0, 0, 0
        )

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate):
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_position_target_local_ned_send(
            now_ms - self.system_boot_ms, self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, VELOCITY_ONLY_MASK,
            0, 0, 0, float(vx), float(vy), float(vz), 0, 0, 0, 0, float(yaw_rate)
        )
