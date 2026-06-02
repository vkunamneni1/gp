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

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.startup_time = time.time()
        self.state = "WAIT"
        print("[CTRL] MINIMAL SCRIPT. WAITING 5 SECONDS ON GROUND...", flush=True)

    def update(self):
        elapsed = time.time() - self.startup_time

        if self.state == "WAIT":
            if elapsed > 5.0:
                print("[CTRL] 5 SECONDS PASSED. TAKING OFF NOW!", flush=True)
                self.state = "TAKEOFF"
                self.takeoff_start = time.time()
            else:
                if int(elapsed * 10) % 10 == 0:
                    print(f"[CTRL] Waiting... {5.0 - elapsed:.1f}s", flush=True)

        elif self.state == "TAKEOFF":
            # Go UP for 3 seconds
            t_elapsed = time.time() - self.takeoff_start
            if t_elapsed < 3.0:
                self._send_velocity_ned(0.0, 0.0, -3.0, 0.0)
            else:
                self.state = "MOVE_FORWARD"

        elif self.state == "MOVE_FORWARD":
            print("[CTRL] MOVING FORWARD! (vx = 5.0, vz = 0.0)", flush=True)
            self._send_velocity_ned(5.0, 0.0, 0.0, 0.0)

        time.sleep(0.02) # 50Hz

    def arm(self):
        print("[CTRL] ARMING CALLED FROM MAIN!", flush=True)
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

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
