#
# AI Grand Prix — Autonomous Drone Racing Client
# ================================================
# Connects to the simulator, arms the drone, and runs the
# autonomous navigation control loop until the race is complete.
#

import sys
import time
import signal

from setup import setup_components

# Modify these properties if you want to run the server remotely
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# Time since sim started ms
system_boot_ms = int(time.time() * 1000)

# Shared data between all components (telemetry, vision, race state)
shared_data = {}

# Graceful shutdown flag
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[MAIN] Shutdown requested (Ctrl+C)...", flush=True)
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

# Setup all components
print("=" * 60, flush=True)
print("  AI GRAND PRIX — Autonomous Drone Pilot", flush=True)
print("=" * 60, flush=True)
print(f"Connecting to simulator at {SIM_SERVER_UDP_IP}:{SIM_SERVER_UDP_PORT}...", flush=True)

components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']

print("", flush=True)
print("Resetting simulator to clear any stuck states...", flush=True)
controller.send_sim_reset_command()
time.sleep(1.0)

print("Arming drone...", flush=True)
controller.arm()
time.sleep(0.5)  # brief delay to let arm command process

print("Starting autonomous control loop...", flush=True)
print("=" * 60, flush=True)

try:
    while not shutdown_requested:
        controller.update()

        # Check if race is finished
        race = shared_data.get('race_status', {})
        if race.get('race_finished', False):
            print("[MAIN] Race finished! Exiting in 3 seconds...", flush=True)
            time.sleep(3.0)
            break

except KeyboardInterrupt:
    print("\n[MAIN] Interrupted.", flush=True)
except Exception as e:
    print(f"\n[MAIN] Error: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Clean shutdown
print("[MAIN] Shutting down...", flush=True)
ts_loop.get_thread_for_join().join(timeout=2.0)
mavlink_rx.get_thread_for_join().join(timeout=2.0)
vision_rx.get_thread_for_join().join(timeout=2.0)

print("[MAIN] Client exited!", flush=True)
