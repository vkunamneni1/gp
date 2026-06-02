import time
import math
import numpy as np
from pymavlink import mavutil

# Ignore body rates, use attitude quaternion + thrust
ATTITUDE_MODE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE | \
                     mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE | \
                     mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE

def euler_to_quat(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.state = "WAIT_RACE"
        self.state_start = time.time()
        self.start_yaw = None
        print("[CTRL] ANGLE FLIGHT MODE SCRIPT INITIALIZED", flush=True)

    def update(self):
        now = time.time()
        
        # Capture initial yaw to maintain heading
        att = self.data.get('attitude', {})
        current_yaw = att.get('yaw', 0.0)
        if self.start_yaw is None:
            self.start_yaw = current_yaw

        if self.state == "WAIT_RACE":
            race = self.data.get('race_status', {})
            if race.get('race_started', False):
                print("[CTRL] RACE STARTED! TAKING OFF!", flush=True)
                self.state = "TAKEOFF"
                self.state_start = now
            else:
                # Send 0 angle, tiny thrust to keep alive but not move
                self._send_attitude(0.0, 0.0, self.start_yaw, 0.01)
                if int(now * 10) % 20 == 0:
                    print("[CTRL] Waiting for race start (Angle Mode)...", flush=True)

        elif self.state == "TAKEOFF":
            # Send 0 angle, large thrust to climb
            t_elapsed = now - self.state_start
            if t_elapsed < 1.5:
                self._send_attitude(0.0, 0.0, self.start_yaw, 0.8)
            else:
                print("[CTRL] MOVING FORWARD!", flush=True)
                self.state = "MOVE_FORWARD"

        elif self.state == "MOVE_FORWARD":
            # Pitch down 20 degrees (0.35 rad) to fly forward, moderate thrust
            self._send_attitude(0.0, 0.35, self.start_yaw, 0.6)
            if int(now * 10) % 20 == 0:
                print("[CTRL] FLYING FORWARD! (Pitch 20deg)", flush=True)

        time.sleep(0.02)

    def arm(self):
        print("[CTRL] ARMING CALLED FROM MAIN!", flush=True)
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

    def _send_attitude(self, roll, pitch, yaw, thrust):
        now_ms = int(time.time() * 1000)
        q = euler_to_quat(roll, pitch, yaw)
        self.sim_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            ATTITUDE_MODE_MASK,
            q,
            0, 0, 0, # Ignored rates
            float(thrust)
        )
