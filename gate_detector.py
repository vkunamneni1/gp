"""
Gate Detection & Pose Estimation for AI Grand Prix
=====================================================
Detects racing gates from FPV camera frames using color segmentation,
estimates 3D pose using solvePnP, and transforms to body NED frame.

Camera specs (from tech spec):
  - Resolution: 640x360
  - Intrinsics: fx=fy=320, cx=320, cy=180
  - No lens distortion
  - Tilted 20° upward from body frame

Gate specs:
  - Inner opening: 1500mm x 1500mm (1.5m x 1.5m)
  - Outer boundary: 2700mm x 2700mm
"""

import math
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Camera intrinsics (from tech spec)
# ---------------------------------------------------------------------------
CAMERA_MATRIX = np.array([
    [320.0,   0.0, 320.0],
    [  0.0, 320.0, 180.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)  # no distortion

# Camera tilt: 20° upward. In NED the camera X-axis points forward,
# and the tilt rotates around the body Y-axis (pitch up = negative rotation
# in NED convention, but the camera looks UP so the rotation brings the
# camera Z-axis upward).
CAMERA_TILT_DEG = 20.0
CAMERA_TILT_RAD = math.radians(CAMERA_TILT_DEG)

# Rotation matrix: body_from_camera  (rotate camera frame into body NED)
# Camera convention: X-right, Y-down, Z-forward (OpenCV)
# Body NED: X-forward, Y-right, Z-down
# The camera is mounted pointing forward but tilted 20° up.
#
# We build R_body_from_cam to transform a point in camera coords to body coords.
_cos_t = math.cos(CAMERA_TILT_RAD)
_sin_t = math.sin(CAMERA_TILT_RAD)

# OpenCV camera: Z forward, X right, Y down
# Body NED:      X forward, Y right, Z down
# Base transform (no tilt): body_X = cam_Z, body_Y = cam_X, body_Z = cam_Y
# Then apply 20° pitch-up tilt around body Y axis
R_CAM_TO_BODY_BASE = np.array([
    [0.0, 0.0, 1.0],   # body_X = cam_Z (forward)
    [1.0, 0.0, 0.0],   # body_Y = cam_X (right)
    [0.0, 1.0, 0.0],   # body_Z = cam_Y (down)
], dtype=np.float64)

# Pitch-up rotation around body Y axis by +20°
R_PITCH_UP = np.array([
    [ _cos_t, 0.0, _sin_t],
    [    0.0, 1.0,    0.0],
    [-_sin_t, 0.0, _cos_t],
], dtype=np.float64)

R_CAM_TO_BODY = R_PITCH_UP @ R_CAM_TO_BODY_BASE

# ---------------------------------------------------------------------------
# Gate 3D model points (inner opening corners in gate-local coords)
# Gate is 1.5m x 1.5m. We place the gate center at the origin.
# Points ordered: top-left, top-right, bottom-right, bottom-left
# In gate-local frame: X-right, Y-up, Z-out-of-gate
# ---------------------------------------------------------------------------
GATE_HALF_W = 0.75   # 1.5m / 2
GATE_HALF_H = 0.75

GATE_3D_POINTS = np.array([
    [-GATE_HALF_W,  GATE_HALF_H, 0.0],   # top-left
    [ GATE_HALF_W,  GATE_HALF_H, 0.0],   # top-right
    [ GATE_HALF_W, -GATE_HALF_H, 0.0],   # bottom-right
    [-GATE_HALF_W, -GATE_HALF_H, 0.0],   # bottom-left
], dtype=np.float64)

# ---------------------------------------------------------------------------
# HSV color ranges to try for gate detection
# We try multiple ranges and pick the best detection.
# VQ1 is described as "high-contrast, desaturated" environment so gates
# should pop. Common gate colors: cyan/blue, orange, green, magenta, red.
# ---------------------------------------------------------------------------
HSV_RANGES = [
    # (name, lower_hsv, upper_hsv)
    ("cyan",       np.array([80, 80, 80]),    np.array([100, 255, 255])),
    ("blue",       np.array([100, 80, 80]),   np.array([130, 255, 255])),
    ("green",      np.array([35, 80, 80]),    np.array([85, 255, 255])),
    ("orange",     np.array([5, 100, 100]),   np.array([25, 255, 255])),
    ("red_low",    np.array([0, 100, 100]),   np.array([10, 255, 255])),
    ("red_high",   np.array([160, 100, 100]), np.array([180, 255, 255])),
    ("magenta",    np.array([140, 80, 80]),   np.array([170, 255, 255])),
    ("yellow",     np.array([20, 80, 80]),    np.array([40, 255, 255])),
    ("white",      np.array([0, 0, 180]),     np.array([180, 50, 255])),
    ("bright_any", np.array([0, 100, 150]),   np.array([180, 255, 255])),
]


class GateDetector:
    """
    Detects racing gates in FPV camera images using color segmentation
    and estimates 3D pose with solvePnP.
    """

    def __init__(self):
        # Locked-in HSV range once we find one that works
        self.locked_hsv_index = None
        self.detection_counts = {}  # track which ranges work best

        # Smoothing: exponential moving average of gate position
        self.smooth_tvec = None
        self.smooth_alpha = 0.4   # weight for new measurements

        # Last known detection for fallback
        self.last_gate_body_ned = None
        self.last_detection_time = 0
        self.frames_since_detection = 0

        # Morphological kernel for noise removal
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def detect(self, image, timestamp=0):
        """
        Main detection entry point.

        Args:
            image: BGR image from camera (640x360)
            timestamp: frame timestamp for tracking

        Returns:
            dict with keys:
                'detected': bool
                'corners': 4x2 array of pixel corners or None
                'center_px': (cx, cy) gate center in pixels or None
                'gate_area': area of gate in pixels or None
                'tvec_cam': translation vector in camera frame or None
                'tvec_body_ned': translation vector in body NED or None
                'distance': distance to gate in meters or None
                'pixel_error': (ex, ey) error from image center or None
        """
        result = {
            'detected': False,
            'corners': None,
            'center_px': None,
            'gate_area': None,
            'tvec_cam': None,
            'tvec_body_ned': None,
            'distance': None,
            'pixel_error': None,
        }

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        best_corners = None
        best_area = 0
        best_idx = None

        # If we've locked in a color, try that first
        if self.locked_hsv_index is not None:
            indices = [self.locked_hsv_index]
        else:
            indices = range(len(HSV_RANGES))

        for idx in indices:
            name, lower, upper = HSV_RANGES[idx]
            corners, area = self._detect_with_range(hsv, lower, upper)
            if corners is not None and area > best_area:
                best_corners = corners
                best_area = area
                best_idx = idx

        # If locked range failed, try all ranges
        if best_corners is None and self.locked_hsv_index is not None:
            for idx in range(len(HSV_RANGES)):
                if idx == self.locked_hsv_index:
                    continue
                name, lower, upper = HSV_RANGES[idx]
                corners, area = self._detect_with_range(hsv, lower, upper)
                if corners is not None and area > best_area:
                    best_corners = corners
                    best_area = area
                    best_idx = idx

        if best_corners is None:
            self.frames_since_detection += 1
            return result

        # Update lock-in tracking
        if best_idx is not None:
            self.detection_counts[best_idx] = self.detection_counts.get(best_idx, 0) + 1
            # Lock in after 10 consistent detections
            if self.detection_counts[best_idx] >= 10 and self.locked_hsv_index is None:
                self.locked_hsv_index = best_idx
                name = HSV_RANGES[best_idx][0]
                print(f"[VISION] Locked gate color: {name}", flush=True)

        # Compute gate center in pixel space
        center_px = np.mean(best_corners, axis=0)

        # Pixel error from image center
        img_cx, img_cy = 320.0, 180.0
        pixel_error = (center_px[0] - img_cx, center_px[1] - img_cy)

        # 3D pose estimation
        corners_2d = best_corners.astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            GATE_3D_POINTS, corners_2d, CAMERA_MATRIX, DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            # Fallback to default solvePnP
            success, rvec, tvec = cv2.solvePnP(
                GATE_3D_POINTS, corners_2d, CAMERA_MATRIX, DIST_COEFFS
            )

        if not success:
            self.frames_since_detection += 1
            return result

        tvec = tvec.flatten()
        distance = float(np.linalg.norm(tvec))

        # Transform to body NED
        tvec_body_ned = R_CAM_TO_BODY @ tvec

        # Smooth the estimate
        if self.smooth_tvec is None:
            self.smooth_tvec = tvec_body_ned.copy()
        else:
            self.smooth_tvec = (self.smooth_alpha * tvec_body_ned +
                                (1.0 - self.smooth_alpha) * self.smooth_tvec)

        self.last_gate_body_ned = self.smooth_tvec.copy()
        self.last_detection_time = timestamp
        self.frames_since_detection = 0

        result['detected'] = True
        result['corners'] = best_corners
        result['center_px'] = (float(center_px[0]), float(center_px[1]))
        result['gate_area'] = float(best_area)
        result['tvec_cam'] = tvec
        result['tvec_body_ned'] = self.smooth_tvec.copy()
        result['distance'] = distance
        result['pixel_error'] = pixel_error

        return result

    def _detect_with_range(self, hsv_img, lower, upper):
        """
        Try to detect a gate quadrilateral using a specific HSV range.

        Returns:
            (corners, area) or (None, 0)
            corners is a 4x2 numpy array ordered: TL, TR, BR, BL
        """
        mask = cv2.inRange(hsv_img, lower, upper)

        # Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0

        # We want to find a roughly square/rectangular gate shape.
        # Strategy: look for the largest contour that can be approximated
        # to ~4 corners, or find the gate as a hollow rectangle (two nested
        # contours forming a frame).

        best_corners = None
        best_area = 0

        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:  # check top 5
            area = cv2.contourArea(contour)
            if area < 200:  # too small to be a gate
                continue

            # Try approximating to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                # Verify it's roughly convex and square-ish
                if self._is_valid_gate(corners):
                    ordered = self._order_corners(corners)
                    if area > best_area:
                        best_corners = ordered
                        best_area = area
            elif len(approx) > 4:
                # Try to find the bounding quadrilateral
                # Use minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = box.astype(np.float32)
                # Check aspect ratio
                w, h = rect[1]
                if w > 0 and h > 0:
                    aspect = max(w, h) / min(w, h)
                    if aspect < 2.5:  # roughly square
                        ordered = self._order_corners(box)
                        if area > best_area:
                            best_corners = ordered
                            best_area = area

        # If we couldn't find a good quad from single contours,
        # try using the convex hull of all gate-colored pixels
        if best_corners is None and len(contours) >= 2:
            # Merge nearby contours
            all_points = np.vstack([c for c in contours[:10] if cv2.contourArea(c) > 50])
            if len(all_points) > 4:
                hull = cv2.convexHull(all_points)
                area = cv2.contourArea(hull)
                if area > 500:
                    rect = cv2.minAreaRect(hull)
                    box = cv2.boxPoints(rect)
                    box = box.astype(np.float32)
                    w, h = rect[1]
                    if w > 0 and h > 0:
                        aspect = max(w, h) / min(w, h)
                        if aspect < 2.5:
                            best_corners = self._order_corners(box)
                            best_area = area

        return best_corners, best_area

    def _is_valid_gate(self, corners):
        """Check if 4 corners form a roughly valid gate shape."""
        # Check convexity
        hull = cv2.convexHull(corners, returnPoints=False)
        if len(hull) != 4:
            return False

        # Check aspect ratio of bounding rect
        rect = cv2.minAreaRect(corners)
        w, h = rect[1]
        if w == 0 or h == 0:
            return False
        aspect = max(w, h) / min(w, h)
        if aspect > 3.0:
            return False

        return True

    def _order_corners(self, corners):
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.
        """
        # Sort by y-coordinate (top to bottom), then by x within each group
        corners = corners.reshape(4, 2)
        # Sum of coordinates: smallest sum = top-left, largest = bottom-right
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).flatten()

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(s)]     # top-left
        ordered[2] = corners[np.argmax(s)]     # bottom-right
        ordered[1] = corners[np.argmin(diff)]  # top-right
        ordered[3] = corners[np.argmax(diff)]  # bottom-left

        return ordered

    def get_last_known_gate(self):
        """
        Get the last known gate position in body NED.
        Returns None if no recent detection.
        """
        if self.last_gate_body_ned is not None and self.frames_since_detection < 30:
            return self.last_gate_body_ned.copy()
        return None
