"""
Gate Detection for AI Grand Prix
===================================
Detects racing gates from FPV camera frames using color segmentation.
Returns gate center in pixel space, gate area, and optionally 3D pose.

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

# Image center
IMG_CX = 320.0
IMG_CY = 180.0

# Gate 3D model for solvePnP (1.5m x 1.5m inner opening)
GATE_HALF = 0.75
GATE_3D_POINTS = np.array([
    [-GATE_HALF,  GATE_HALF, 0.0],   # top-left
    [ GATE_HALF,  GATE_HALF, 0.0],   # top-right
    [ GATE_HALF, -GATE_HALF, 0.0],   # bottom-right
    [-GATE_HALF, -GATE_HALF, 0.0],   # bottom-left
], dtype=np.float64)

# Camera tilt transform (20° upward)
CAMERA_TILT_RAD = math.radians(20.0)
_ct = math.cos(CAMERA_TILT_RAD)
_st = math.sin(CAMERA_TILT_RAD)

# Camera (OpenCV: X-right, Y-down, Z-forward) → Body NED (X-fwd, Y-right, Z-down)
# Then rotate by 20° pitch-up around body Y
R_CAM_TO_BODY = np.array([
    [ _ct, 0.0, 1.0],   # simplified: cam_Z→body_X with pitch rotation
    [ 0.0, 1.0, 0.0],   # cam_X→body_Y
    [-_st, 0.0, 1.0],   # cam_Y→body_Z with pitch rotation
], dtype=np.float64)

# More correct version:
# Base: body = [cam_Z, cam_X, cam_Y]
# Pitch up by 20° around body Y:
R_BASE = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=np.float64)
R_PITCH = np.array([
    [ _ct, 0, _st],
    [  0,  1,   0],
    [-_st, 0, _ct],
], dtype=np.float64)
R_CAM_TO_BODY = R_PITCH @ R_BASE

# ---------------------------------------------------------------------------
# HSV color ranges for gate detection.
# VQ1 = "high-contrast, desaturated" environment. Gates should pop.
# ---------------------------------------------------------------------------
HSV_RANGES = [
    # (name, lower_hsv, upper_hsv)
    ("cyan",       np.array([80, 60, 60]),    np.array([105, 255, 255])),
    ("blue",       np.array([100, 60, 60]),   np.array([130, 255, 255])),
    ("green",      np.array([35, 60, 60]),    np.array([85, 255, 255])),
    ("orange",     np.array([5, 80, 80]),     np.array([25, 255, 255])),
    ("red_low",    np.array([0, 80, 80]),     np.array([10, 255, 255])),
    ("red_high",   np.array([160, 80, 80]),   np.array([180, 255, 255])),
    ("magenta",    np.array([140, 60, 60]),   np.array([170, 255, 255])),
    ("yellow",     np.array([20, 60, 60]),    np.array([40, 255, 255])),
    ("white_bright", np.array([0, 0, 200]),   np.array([180, 40, 255])),
    ("neon_any",   np.array([0, 120, 150]),   np.array([180, 255, 255])),
]


class GateDetector:
    """Detects racing gates using color segmentation."""

    def __init__(self):
        self.locked_hsv_index = None
        self.detection_counts = {}
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def detect(self, image):
        """
        Detect a gate in the image.

        Returns dict:
            detected: bool
            center_px: (cx, cy) gate center in pixels
            pixel_error: (ex, ey) offset from image center
            gate_area: area in pixels
            distance: estimated distance (meters) or None
        """
        result = {
            'detected': False,
            'center_px': None,
            'pixel_error': None,
            'gate_area': 0,
            'distance': None,
        }

        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except Exception:
            return result

        best_center = None
        best_area = 0
        best_corners = None
        best_idx = None

        # If locked, try that first
        if self.locked_hsv_index is not None:
            indices_to_try = [self.locked_hsv_index]
        else:
            indices_to_try = list(range(len(HSV_RANGES)))

        for idx in indices_to_try:
            name, lower, upper = HSV_RANGES[idx]
            center, area, corners = self._detect_gate_in_range(hsv, lower, upper)
            if center is not None and area > best_area:
                best_center = center
                best_area = area
                best_corners = corners
                best_idx = idx

        # If locked range failed, try all
        if best_center is None and self.locked_hsv_index is not None:
            for idx in range(len(HSV_RANGES)):
                if idx == self.locked_hsv_index:
                    continue
                name, lower, upper = HSV_RANGES[idx]
                center, area, corners = self._detect_gate_in_range(hsv, lower, upper)
                if center is not None and area > best_area:
                    best_center = center
                    best_area = area
                    best_corners = corners
                    best_idx = idx

        if best_center is None:
            return result

        # Lock-in after consistent detections
        if best_idx is not None:
            self.detection_counts[best_idx] = self.detection_counts.get(best_idx, 0) + 1
            if self.detection_counts[best_idx] >= 8 and self.locked_hsv_index is None:
                self.locked_hsv_index = best_idx
                print(f"[VISION] Locked gate color: {HSV_RANGES[best_idx][0]}", flush=True)

        # Pixel error from image center
        px_err_x = best_center[0] - IMG_CX
        px_err_y = best_center[1] - IMG_CY

        # Distance estimate from gate area
        # Gate inner opening = 1.5m x 1.5m = 2.25 m²
        # At distance d, gate appears as (1.5 * fx / d) x (1.5 * fy / d) pixels
        # area_px ≈ (1.5 * 320 / d)² = (480/d)² = 230400/d²
        # d ≈ sqrt(230400 / area_px)
        distance = None
        if best_area > 100:
            distance = math.sqrt(230400.0 / best_area)

        # Try solvePnP if we have good corners
        if best_corners is not None and len(best_corners) == 4:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    GATE_3D_POINTS, best_corners.astype(np.float64),
                    CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if success:
                    d_pnp = float(np.linalg.norm(tvec))
                    if 0.5 < d_pnp < 100:  # sanity check
                        distance = d_pnp
            except Exception:
                pass  # Fall back to area-based estimate

        result['detected'] = True
        result['center_px'] = (float(best_center[0]), float(best_center[1]))
        result['pixel_error'] = (float(px_err_x), float(px_err_y))
        result['gate_area'] = float(best_area)
        result['distance'] = distance

        return result

    def _detect_gate_in_range(self, hsv_img, lower, upper):
        """
        Detect a gate-like quadrilateral in a specific HSV range.

        Returns:
            (center, area, corners) or (None, 0, None)
            center: (cx, cy) center of detected gate
            area: area in pixels
            corners: 4x2 array of ordered corners, or None
        """
        mask = cv2.inRange(hsv_img, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, None

        best_center = None
        best_area = 0
        best_corners = None

        # Sort by area, try largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area < 300:  # minimum gate size in pixels
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

            if 4 <= len(approx) <= 8:
                # Check aspect ratio with minAreaRect
                rect = cv2.minAreaRect(contour)
                w, h = rect[1]
                if w > 0 and h > 0:
                    aspect = max(w, h) / min(w, h)
                    if aspect > 3.5:
                        continue  # too elongated, not a gate

                # Get center via moments
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = rect[0]

                if area > best_area:
                    best_center = (cx, cy)
                    best_area = area

                    # Try to get ordered corners
                    if len(approx) == 4:
                        corners = approx.reshape(4, 2).astype(np.float32)
                        best_corners = self._order_corners(corners)
                    else:
                        # Use minAreaRect corners
                        box = cv2.boxPoints(rect).astype(np.float32)
                        best_corners = self._order_corners(box)

        # If no single contour worked, try combining nearby contours
        if best_center is None:
            # Filter contours with minimum area
            valid = [c for c in contours[:10] if cv2.contourArea(c) > 80]
            if len(valid) >= 2:
                try:
                    all_points = np.vstack(valid)
                    hull = cv2.convexHull(all_points)
                    area = cv2.contourArea(hull)
                    if area > 500:
                        rect = cv2.minAreaRect(hull)
                        w, h = rect[1]
                        if w > 0 and h > 0 and max(w, h) / min(w, h) < 3.5:
                            M = cv2.moments(hull)
                            if M["m00"] > 0:
                                cx = M["m10"] / M["m00"]
                                cy = M["m01"] / M["m00"]
                                best_center = (cx, cy)
                                best_area = area
                                box = cv2.boxPoints(rect).astype(np.float32)
                                best_corners = self._order_corners(box)
                except (ValueError, cv2.error):
                    pass  # Safely handle empty arrays

        return best_center, best_area, best_corners

    def _order_corners(self, corners):
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        corners = corners.reshape(4, 2)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).flatten()
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(s)]
        ordered[2] = corners[np.argmax(s)]
        ordered[1] = corners[np.argmin(diff)]
        ordered[3] = corners[np.argmax(diff)]
        return ordered
