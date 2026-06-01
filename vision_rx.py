"""
Vision Receiver — Receives FPV camera frames and runs gate detection.
=====================================================================
Receives JPEG frames via UDP (port 5600), reassembles multi-packet
frames, decodes with OpenCV, and runs the GateDetector pipeline.
Detection results are stored in shared_data['vision_detection'].
"""

import socket
import struct
import threading
import time

import cv2
import numpy as np

from gate_detector import GateDetector

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600


class VisionRX:

    def __init__(self, data):
        self.data = data
        self.gate_detector = GateDetector()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        # Initialize vision detection data
        self.data['vision_detection'] = {
            'detected': False,
            'corners': None,
            'center_px': None,
            'gate_area': None,
            'tvec_cam': None,
            'tvec_body_ned': None,
            'distance': None,
            'pixel_error': None,
        }

        self.thread = threading.Thread(
            target=self._vision_loop,
            daemon=False
        )
        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _vision_loop(self):
        header_format = "<IHHIIQ"
        header_sz = struct.calcsize(header_format)
        frames = {}  # frame_id -> received associated frame data

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)  # 1 second timeout so we can check is_running
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("[VISION] Listening for camera frames on port 5600...", flush=True)

        while self.is_running:
            try:
                packet, addr = sock.recvfrom(65536)  # max UDP size
            except socket.timeout:
                continue
            except OSError:
                break

            header = packet[:header_sz]
            payload = packet[header_sz:]

            # frame_id - identifier for this vision frame
            # chunk_id - identifier for this chunk packet of data of this frame
            # total_chunks - total number of chunk packets that make up this frame
            # jpeg_size - full size of jpeg data
            # payload_size - size of this packet
            # sim_time_ns - frame's epoch timestamp in ns on the server
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(header_format, header)

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns
                }

            frames[frame_id]["chunks"][chunk_id] = payload

            # Check if frame is complete
            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()

                frame_complete = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        frame_complete = False
                        continue
                    jpeg_bytes.extend(frames[frame_id]["chunks"][i])

                if not frame_complete:
                    del frames[frame_id]
                    continue

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    self.process_frame(frame_id, image, sim_time_ns)

                del frames[frame_id]

                # Clean up old incomplete frames (prevent memory leak)
                stale = [fid for fid in frames if fid < frame_id - 10]
                for fid in stale:
                    del frames[fid]

        sock.close()

    def process_frame(self, frame_id, img, timestamp_ns=0):
        """
        Process an FPV camera frame: run gate detection and store results.
        """
        self.frame_count += 1

        # Run gate detection
        result = self.gate_detector.detect(img, timestamp=timestamp_ns)

        # Store results in shared data for the controller
        self.data['vision_detection'] = result

        # FPS tracking
        now = time.time()
        if now - self.last_fps_time >= 5.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

            if result['detected']:
                d = result.get('distance', 0)
                px_err = result.get('pixel_error', (0, 0))
                print(f"[VISION] {self.fps:.1f} FPS | Gate detected: "
                      f"dist={d:.1f}m px_err=({px_err[0]:.0f},{px_err[1]:.0f})",
                      flush=True)
            else:
                print(f"[VISION] {self.fps:.1f} FPS | No gate detected", flush=True)