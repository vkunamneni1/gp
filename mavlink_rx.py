"""
MAVLink Receiver — Stores all telemetry into shared_data for the controller.
============================================================================
Receives and parses: HEARTBEAT, ATTITUDE, LOCAL_POSITION_NED, ODOMETRY,
HIGHRES_IMU, COLLISION, RACE_STATUS, TRACK_DATA, ACTUATOR_OUTPUT_STATUS.
"""

import struct
import time
import threading
import math

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2


class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False

        self.track_chunks = {}
        self.expected_num_track_chunks = {}

        # Debug: track which message types we receive from the sim
        self._seen_msg_types = set()

        # Initialize all data fields so the controller never gets KeyError
        self.data['armed'] = False
        self.data['attitude'] = {
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'rollspeed': 0.0, 'pitchspeed': 0.0, 'yawspeed': 0.0,
            'time_boot_ms': 0
        }
        self.data['local_position'] = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'time_boot_ms': 0
        }
        self.data['odometry'] = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'rollspeed': 0.0, 'pitchspeed': 0.0, 'yawspeed': 0.0,
            'time_usec': 0, 'reset_counter': 0
        }
        self.data['imu'] = {
            'xacc': 0.0, 'yacc': 0.0, 'zacc': 0.0,
            'xgyro': 0.0, 'ygyro': 0.0, 'zgyro': 0.0,
            'time_usec': 0
        }
        self.data['collision'] = {
            'active': False, 'id': 0, 'threat_level': 0,
            'impulse': 0.0, 'timestamp': 0
        }
        self.data['race_status'] = {
            'sim_boot_time_ms': 0,
            'race_start_boot_time_ms': -1,
            'race_finish_time_ns': -1,
            'active_gate_index': 0,
            'last_gate_race_time': 0,
            'race_started': False,
            'race_finished': False,
        }
        self.data['track'] = {
            'gates': [],
            'num_gates': 0,
            'received': False,
        }
        self.data['motors'] = {
            'front_left': 0.0, 'front_right': 0.0,
            'back_left': 0.0, 'back_right': 0.0,
            'time_usec': 0
        }
        self.data['heartbeat_count'] = 0
        self.data['has_position'] = False

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(
            target=rx.mavlink_receive_loop,
            daemon = False
        )
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.
        """
        while self.is_running:

            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print('WARNING: ConnectionResetError was thrown. No longer listening to MAVLink port.')
                return

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()

            if msg_type == "BAD_DATA":
                continue

            # Debug: log first occurrence of each message type
            if msg_type not in self._seen_msg_types:
                self._seen_msg_types.add(msg_type)
                print(f"[MAVLINK] First received: {msg_type}", flush=True)

            # --------------------------------------------------------------------------------------
            # HEARTBEAT
            # --------------------------------------------------------------------------------------
            if msg_type == "HEARTBEAT":
                self.on_heartbeat(msg)

            # --------------------------------------------------------------------------------------
            # TIMESYNC
            # --------------------------------------------------------------------------------------
            elif msg_type == "TIMESYNC":
                self.on_timesync(msg)

            # --------------------------------------------------------------------------------------
            # ATTITUDE
            # --------------------------------------------------------------------------------------
            elif msg_type == "ATTITUDE":
                self.on_attitude(msg)

            # --------------------------------------------------------------------------------------
            # LOCAL_POSITION_NED
            # --------------------------------------------------------------------------------------
            elif msg_type == "LOCAL_POSITION_NED":
                self.on_local_position_ned(msg)

            # --------------------------------------------------------------------------------------
            # ODOMETRY
            # --------------------------------------------------------------------------------------
            elif msg_type == "ODOMETRY":
                self.on_odometry(msg)

            # --------------------------------------------------------------------------------------
            # HIGHRES_IMU
            # --------------------------------------------------------------------------------------
            elif msg_type == "HIGHRES_IMU":
                self.on_highres_imu(msg)

            # --------------------------------------------------------------------------------------
            # ENCAPSULATED_DATA
            # --------------------------------------------------------------------------------------
            elif msg_type == "ENCAPSULATED_DATA":
                self.on_encapsulated_data(msg)

            # --------------------------------------------------------------------------------------
            # ACTUATOR_OUTPUT_STATUS
            # --------------------------------------------------------------------------------------
            elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                self.on_actuator_output_status(msg)

            # --------------------------------------------------------------------------------------
            # COLLISION
            # --------------------------------------------------------------------------------------
            elif msg_type == "COLLISION":
                self.on_collision(msg)

            # --------------------------------------------------------------------------------------
            # DATA_TRANSMISSION_HANDSHAKE - Repurposed and used for upcoming 'Track Data' packets
            # --------------------------------------------------------------------------------------
            elif msg.get_type() == "DATA_TRANSMISSION_HANDSHAKE":
                track_data_transfer_id = msg.width
                self.track_chunks[track_data_transfer_id] = {}
                self.expected_num_track_chunks[track_data_transfer_id] = msg.packets

    def on_heartbeat(self, msg):
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        self.data['armed'] = bool(armed)
        self.data['heartbeat_count'] = self.data.get('heartbeat_count', 0) + 1

    def on_timesync(self, msg):
        request_time = msg.ts1
        response_time = msg.tc1

    def on_attitude(self, msg):
        self.data['attitude'] = {
            'roll': msg.roll,
            'pitch': msg.pitch,
            'yaw': msg.yaw,
            'rollspeed': msg.rollspeed,
            'pitchspeed': msg.pitchspeed,
            'yawspeed': msg.yawspeed,
            'time_boot_ms': msg.time_boot_ms,
        }

    def on_local_position_ned(self, msg):
        self.data['local_position'] = {
            'x': msg.x,
            'y': msg.y,
            'z': msg.z,
            'vx': msg.vx,
            'vy': msg.vy,
            'vz': msg.vz,
            'time_boot_ms': msg.time_boot_ms,
        }
        self.data['has_position'] = True

    def on_odometry(self, msg):
        self.data['odometry'] = {
            'x': msg.x, 'y': msg.y, 'z': msg.z,
            'qw': msg.q[0], 'qx': msg.q[1], 'qy': msg.q[2], 'qz': msg.q[3],
            'vx': msg.vx, 'vy': msg.vy, 'vz': msg.vz,
            'rollspeed': msg.rollspeed,
            'pitchspeed': msg.pitchspeed,
            'yawspeed': msg.yawspeed,
            'time_usec': msg.time_usec,
            'reset_counter': msg.reset_counter,
        }

    def on_highres_imu(self, msg):
        self.data['imu'] = {
            'xacc': msg.xacc, 'yacc': msg.yacc, 'zacc': msg.zacc,
            'xgyro': msg.xgyro, 'ygyro': msg.ygyro, 'zgyro': msg.zgyro,
            'time_usec': msg.time_usec,
        }

    def on_encapsulated_data(self, msg):
        if msg:
            raw_payload = bytes(msg.data)
            data_type = raw_payload[0]

            if int(data_type) == ENCAPSULATED_RACE_STATUS_MSG_ID:
                self.on_race_status(msg)
            elif int(data_type) == ENCAPSULATED_TRACK_INFO_MSG_ID:
                self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw_payload = bytes(msg.data)
        # data_type - ID of this message
        # sim_boot_time_ms - elapsed ms on server since sim boot
        # race_start_boot_time_ms - elapsed ms on server since sim boot when race started. None or < 0 if race has not started
        # race_finish_time_ns - elapsed ns on server since sim boot when race finished. None or < 0 if race is ongoing
        # active_gate_index - current index of target race gate
        # last_gate_race_time - race time in seconds when last gate was passed
        data_type, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns, active_gate_index, last_gate_race_time = struct.unpack_from(
            "<BQqqIq", raw_payload)

        race_started = race_start_boot_time_ms is not None and race_start_boot_time_ms > 0
        race_finished = race_finish_time_ns is not None and race_finish_time_ns > 0

        prev_gate = self.data['race_status'].get('active_gate_index', 0)
        if active_gate_index != prev_gate and race_started:
            print(f"[RACE] Gate {prev_gate} passed! Now targeting gate {active_gate_index}. "
                  f"Last gate time: {last_gate_race_time}", flush=True)

        if race_finished and not self.data['race_status'].get('race_finished', False):
            elapsed_s = race_finish_time_ns / 1e9
            print(f"[RACE] *** RACE FINISHED! *** Time: {elapsed_s:.2f}s", flush=True)

        self.data['race_status'] = {
            'sim_boot_time_ms': sim_boot_time_ms,
            'race_start_boot_time_ms': race_start_boot_time_ms,
            'race_finish_time_ns': race_finish_time_ns,
            'active_gate_index': active_gate_index,
            'last_gate_race_time': last_gate_race_time,
            'race_started': race_started,
            'race_finished': race_finished,
        }

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)
        # header:
        #   data_type - ID of this message
        #   transfer_id - ID of the group of packets this chunk belongs to
        data_type, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self.expected_num_track_chunks:
            return
        raw_payload = raw_payload[3:]
        self.track_chunks[transfer_id][msg.seqnr] = raw_payload
        if len(self.track_chunks[transfer_id]) == self.expected_num_track_chunks[transfer_id]:
            full_payload = bytes()
            for i in range(len(self.track_chunks[transfer_id])):
                full_payload = full_payload + self.track_chunks[transfer_id][i]
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full_payload)

    def on_track_data(self, payload):
        # header:
        #   num_gates - track gate count
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]

        gates = []
        for i in range(num_gates):
            # Gate Info
            #   gate_id - range is 0 - num_gates
            #   position_ned_x, position_ned_y, position_ned_z - Position of gate in NED coordinates
            #   orientation_ned_w, orientation_ned_x, orientation_ned_y, orientation_ned_z - Orientation of gate in NED coordinates
            #   width - gate width in metres
            #   height - gate height in metres
            gate_id, pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z, width, height = struct.unpack_from(
                "<Hfffffffff", payload)
            payload = payload[38:]

            gate_info = {
                'id': gate_id,
                'position': (pos_x, pos_y, pos_z),  # NED
                'orientation': (ori_w, ori_x, ori_y, ori_z),  # quaternion
                'width': width,
                'height': height,
            }
            gates.append(gate_info)

        self.data['track'] = {
            'gates': gates,
            'num_gates': num_gates,
            'received': True,
        }
        print(f"[TRACK] Received track data: {num_gates} gates", flush=True)
        for g in gates:
            print(f"  Gate {g['id']}: pos=({g['position'][0]:.1f}, {g['position'][1]:.1f}, {g['position'][2]:.1f}) "
                  f"size=({g['width']:.1f}x{g['height']:.1f})", flush=True)

    def on_actuator_output_status(self, msg):
        self.data['motors'] = {
            'front_left': msg.actuator[0],
            'front_right': msg.actuator[1],
            'back_left': msg.actuator[2],
            'back_right': msg.actuator[3],
            'time_usec': msg.time_usec,
        }

    def on_collision(self, msg):
        # Collision IDs
        # 1001 - Gate
        # 1002 - Environment
        collision_id = msg.id
        threat_level = msg.threat_level # 1-2 with 2 being higher impact collision
        impact = msg.horizontal_minimum_delta # this is not a delta - it is the impulse magnitude in kg m/s

        collision_type = "GATE" if collision_id == 1001 else "ENVIRONMENT" if collision_id == 1002 else f"UNKNOWN({collision_id})"
        print(f"[COLLISION] {collision_type} threat={threat_level} impulse={impact:.2f}", flush=True)

        self.data['collision'] = {
            'active': True,
            'id': collision_id,
            'threat_level': threat_level,
            'impulse': impact,
            'timestamp': time.time(),
        }