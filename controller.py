"""
Autonomous Flight Controller - Velocity Mode with Ground Holding
==============================================================
Fixes:
1. Spams `arm()` while waiting for the race to start to prevent auto-disarm on the ground.
2. Doesn't take off until the race officially starts to prevent drifting across the start line.
3. Uses pure Velocity commands (no Angle mode spinning!).
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

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.state = "WAIT_RACE"
        self.state_start = time.time()
        self.last_arm_time = time.time()
        print("[CTRL] VELOCITY CONTROLLER INITIALIZED. WAITING FOR RACE.", flush=True)
        
        # Force the flight controller back into GUIDED mode (fixes stuck ACRO mode)
        try:
            print("[CTRL] Forcing flight controller to GUIDED mode...", flush=True)
            self.sim_conn.mav.set_mode_send(
                self.sim_conn.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4 # GUIDED mode in ArduPilot
            )
            # Also send via command_long as a fallback
            self.sim_conn.mav.command_long_send(
                self.sim_conn.target_system, self.sim_conn.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4, 0, 0, 0, 0, 0
            )
        except Exception as e:
            print(f"[CTRL] Mode set failed: {e}")

    def update(self):
        now = time.time()

        if self.state == "WAIT_RACE":
            race_started = self.data.get('race_status', {}).get('race_started', False)
            if race_started:
                print("[CTRL] RACE STARTED! TAKING OFF!", flush=True)
                self.state = "TAKEOFF"
                self.state_start = now
            else:
                # Spam arm command every 1 second to prevent auto-disarm while waiting
                if now - self.last_arm_time > 1.0:
                    self.arm()
                    self.last_arm_time = now
                if int(now * 10) % 20 == 0:
                    print("[CTRL] Sitting on the ground waiting for countdown...", flush=True)

        elif self.state == "TAKEOFF":
            # Fly straight up until we reach 2m altitude
            pos = self.data.get('local_position', {})
            current_z = pos.get('z', 0.0) # negative is up
            
            if current_z > -2.0:
                self._send_velocity_ned(0, 0, -3.0, 0)
            else:
                print("[CTRL] Altitude reached! MOVING FORWARD!", flush=True)
                self.state = "NAVIGATE"

        elif self.state == "NAVIGATE":
            # Just fly straight forward for now to prove movement works safely!
            self._send_velocity_ned(5.0, 0, 0, 0)
            if int(now * 10) % 20 == 0:
                print("[CTRL] FLYING FORWARD!", flush=True)

        time.sleep(0.02)

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate):
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_position_target_local_ned_send(
            now_ms - self.system_boot_ms, self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, VELOCITY_ONLY_MASK,
            0, 0, 0, float(vx), float(vy), float(vz), 0, 0, 0, 0, float(yaw_rate)
        )
