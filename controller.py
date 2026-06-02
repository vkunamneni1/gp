import time
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
        self.state = "WAIT"
        self.startup_time = time.time()
        print("[CTRL] Ultimate Bypass Script Initialized.", flush=True)

        # Force GUIDED mode
        try:
            self.sim_conn.mav.set_mode_send(
                self.sim_conn.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4
            )
        except:
            pass

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
            # Wait for race to start (or 5 seconds max)
            race_started = self.data.get('race_status', {}).get('race_started', False)
            if race_started or elapsed > 5.0:
                print(f"[CTRL] Race started (or 5s passed). Arming now!", flush=True)
                self.arm()
                self.state = "ARMING"
                self.takeoff_start = now
            else:
                if int(now * 10) % 20 == 0:
                    print(f"[CTRL] Waiting... {elapsed:.1f}s", flush=True)

        elif self.state == "ARMING":
            if now - self.takeoff_start > 0.5:
                print(f"[CTRL] Armed! Taking off!", flush=True)
                self.state = "TAKEOFF"
                self.takeoff_start = now

        elif self.state == "TAKEOFF":
            t_elapsed = now - self.takeoff_start
            if t_elapsed < 3.0:
                self._send_velocity_ned(0, 0, -3.0, 0)
            else:
                print(f"[CTRL] Altitude reached! Flying forward!", flush=True)
                self.state = "NAVIGATE"

        elif self.state == "NAVIGATE":
            self._send_velocity_ned(5.0, 0, 0, 0)
            if int(now * 10) % 20 == 0:
                print("[CTRL] FLYING FORWARD!", flush=True)

        time.sleep(0.02)

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            31000, # MAVLINK_CMD_SIM_RESET
            0,
            0, 0, 0, 0, 0, 0, 0
        )

    def _send_velocity_ned(self, vx, vy, vz, yaw_rate):
        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_position_target_local_ned_send(
            now_ms - self.system_boot_ms, self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, VELOCITY_ONLY_MASK,
            0, 0, 0, float(vx), float(vy), float(vz), 0, 0, 0, 0, float(yaw_rate)
        )
