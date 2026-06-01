"""
Vision Receiver — Receives FPV camera frames and runs gate detection.
=====================================================================
Crash-protected: gate detection errors don't kill the vision thread.
Adds timestamp to detection results so the controller can check freshness.
"""

import socket
import struct
import threading
import time
import traceback

import cv2
import numpy as np

from gate_detector import GateDetector

SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600


class VisionRX:

    def __init__(self, data):
        self.data = data
        self.gate_detector = GateDetector()
        self.frame_count = 0
        self.total_frames = 0
        self.detect_count = 0
        self.last_fps_time = time.time()

        # Initialize with empty, stale detection
        self.data['vision_detection'] = {
            'detected': False,
            'center_px': None,
            'pixel_error': None,
            'gate_area': 0,
            'distance': None,
            'timestamp': 0,  # wall-clock time of detection
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
        frames = {}

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("[VISION] Listening for camera frames on port 5600...", flush=True)

        while self.is_running:
            try:
                packet, addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break

            header = packet[:header_sz]
            payload = packet[header_sz:]

            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(header_format, header)

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns
                }

            frames[frame_id]["chunks"][chunk_id] = payload

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
                    self.process_frame(frame_id, image)

                del frames[frame_id]

                # Cleanup old incomplete frames
                stale = [fid for fid in frames if fid < frame_id - 10]
                for fid in stale:
                    del frames[fid]

        sock.close()

    def process_frame(self, frame_id, img):
        """Process FPV frame with crash protection."""
        self.total_frames += 1

        try:
            result = self.gate_detector.detect(img)

            # Add wall-clock timestamp
            result['timestamp'] = time.time()

            # Store for controller
            self.data['vision_detection'] = result

            if result['detected']:
                self.detect_count += 1

        except Exception as e:
            # Log error but DON'T crash the vision thread
            if self.total_frames % 100 == 1:  # don't spam
                print(f"[VISION] Detection error (frame {frame_id}): {e}", flush=True)
            # Mark as no detection
            self.data['vision_detection'] = {
                'detected': False,
                'timestamp': time.time(),
            }

        # FPS + detection rate logging
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 5.0:
            elapsed = now - self.last_fps_time
            fps = self.frame_count / elapsed
            det_pct = (self.detect_count / max(self.frame_count, 1)) * 100

            print(f"[VISION] {fps:.1f} FPS | detect rate: {det_pct:.0f}% "
                  f"({self.detect_count}/{self.frame_count} frames)",
                  flush=True)

            self.frame_count = 0
            self.detect_count = 0
            self.last_fps_time = now