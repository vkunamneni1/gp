"""
Autonomous Flight Controller
=============================
Fixes:
1. Arms IMMEDIATELY (like the original example) to prevent "Throttle down please".
2. Takes off IMMEDIATELY to prevent ArduPilot auto-disarm.
3. Hovers at altitude until `race_started` is True (prevents Early Start DQ).
4. Then flies forward!
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
        self.state = "TAKEOFF"
        self.state_start = time.time()
        print("[CTRL] Controller Initialized! Proceeding directly to TAKEOFF.", flush=True)

    def update(self):
        now = time.time()

        if self.state == "TAKEOFF":
            # Fly straight up for 3 seconds to avoid auto-disarm on the ground
            if now - self.state_start < 3.0:
                self._send_velocity_ned(0, 0, -2.5, 0)
            else:
                self.state = "WAIT_RACE"
                print("[CTRL] Airborne! Waiting for race countdown...", flush=True)

        elif self.state == "WAIT_RACE":
            # Hover in place until the race officially starts
            race_started = self.data.get('race_status', {}).get('race_started', False)
            if race_started:
                print("[CTRL] RACE STARTED! Go go go!", flush=True)
                self.state = "NAVIGATE"
            else:
                self._send_velocity_ned(0, 0, 0, 0)
                if int(now * 10) % 20 == 0:
                    print("[CTRL] Hovering at start line...", flush=True)

        elif self.state == "NAVIGATE":
            # Just fly straight forward for now to prove movement works!
            self._send_velocity_ned(4.0, 0, 0, 0)
            if int(now * 10) % 20 == 0:
                print("[CTRL] FLYING FORWARD!", flush=True)

        time.sleep(0.02)

    def arm(self):
        # We leave this here because main.py expects it to exist and calls it immediately!
        print("[CTRL] ARM COMMAND RECEIVED FROM MAIN.PY", flush=True)
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
