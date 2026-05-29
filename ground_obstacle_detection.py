"""
Ground Plane Detection and Obstacle Avoidance Module
Detects ground plane, identifies obstacles, and creates navigable path maps.

FastSAM RGB segmentation is fused with depth-derived obstacle detection on a
background thread so it never stalls the main loop.
"""

import cv2
import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
from ultralytics import FastSAM

# ---------------------------------------------------------------------------
# Tunable constants (mirrored from debug suite)
# ---------------------------------------------------------------------------
PLANE_REFIT_INTERVAL = 10    # Refit ground plane every N calls (0 = every call)
MAX_RANSAC_PTS       = 1500  # Maximum 3-D points fed to RANSAC
RANSAC_ITERS         = 30    # RANSAC iterations
RANSAC_INLIER_THRESH = 0.04  # 4 cm inlier distance
MIN_DISTANCE_M       = 0.35  # Ignore depth < 35 cm (robot chassis / mount clutter)
MAX_DISTANCE_M       = 2.5   # Maximum obstacle detection range

FASTSAM_IMGSZ        = 320   # FastSAM inference resolution (320 = ~4× faster than 640)
FASTSAM_INTERVAL     = 4     # Submit a new frame to FastSAM every N process_frame() calls
FLOOR_UPDATE_INTERVAL = 6    # Relearn floor HSV model every N fuse() calls


class TerrainType(Enum):
    """Classification of terrain types"""
    GROUND   = 0
    OBSTACLE = 1
    UNKNOWN  = 2
    TAG      = 3


@dataclass
class GroundPlane:
    """Represents detected ground plane parameters"""
    normal:     np.ndarray  # 3D normal vector in camera frame (points UP, +Y)
    distance:   float       # Plane equation constant d  (n·p + d = 0)
    confidence: float       # Confidence score [0, 1]
    inliers:    int         # Number of inlier points
    bounds:     tuple       # (min_x, max_x, min_y, max_y) in image coordinates


@dataclass
class Obstacle:
    """Represents a detected obstacle"""
    center:       tuple            # (u, v) pixel coordinates
    position_3d:  np.ndarray      # 3D position in camera frame (metres)
    size:         tuple            # (width, height) pixels
    distance:     float            # Distance from camera in metres
    bearing:      float            # Horizontal angle in radians
    severity:     float            # Obstacle severity [0, 1]
    contour:      Optional[np.ndarray] = None  # Pixel contour for polygon display
    detection_source: str = 'depth'            # 'depth' | 'rgb' | 'depth+rgb'


@dataclass
class PathSegment:
    """Represents a segment of the planned path"""
    start:     np.ndarray  # 3D start position (camera frame)
    end:       np.ndarray  # 3D end position (camera frame)
    width:     float       # Available width in metres
    clearance: float       # Minimum clearance from obstacles in metres
    cost:      float       # Path cost (lower is better)


# ---------------------------------------------------------------------------
# GroundPlaneDetector
# ---------------------------------------------------------------------------
class GroundPlaneDetector:
    """
    Detects ground plane using RANSAC on depth data.
    Key fixes vs original:
      - Restricts point sampling to lower 70% of frame.
      - Normal is forced to point UP (+Y in camera frame) — fixes the sign-
        inversion bug that caused the floor to be detected as an obstacle.
      - Tighter inlier threshold (4 cm).
      - Capped point budget and cached result across PLANE_REFIT_INTERVAL frames.
    """

    def __init__(self,
                 ransac_threshold: float = RANSAC_INLIER_THRESH,
                 min_inliers:      int   = 100,
                 max_iterations:   int   = RANSAC_ITERS):
        self.ransac_threshold = ransac_threshold
        self.min_inliers      = min_inliers
        self.max_iterations   = max_iterations
        self.expected_pitch   = 0.3  # ~17 degrees

        self._plane_cache:       Optional[Tuple[np.ndarray, float]] = None
        self._plane_frame_count: int = 0

    def detect_ground_plane(self,
                             depth_map:         np.ndarray,
                             camera_intrinsics: np.ndarray,
                             tag_masks:         Optional[np.ndarray] = None,
                             force_refit:       bool = False
                             ) -> Optional[GroundPlane]:
        self._plane_frame_count += 1

        if (not force_refit
                and self._plane_cache is not None
                and self._plane_frame_count % PLANE_REFIT_INTERVAL != 0):
            n, d = self._plane_cache
            return GroundPlane(normal=n, distance=d,
                               confidence=1.0, inliers=999, bounds=(0, 0, 0, 0))

        # Downsample to 320×240 for speed
        h, w = depth_map.shape
        if h > 240 or w > 320:
            depth_map = cv2.resize(depth_map, (320, 240),
                                   interpolation=cv2.INTER_NEAREST)
            if tag_masks is not None:
                tag_masks = cv2.resize(tag_masks, (320, 240),
                                       interpolation=cv2.INTER_NEAREST)
            scale_x = 320 / w
            scale_y = 240 / h
            camera_intrinsics = camera_intrinsics.copy()
            camera_intrinsics[0, 0] *= scale_x
            camera_intrinsics[1, 1] *= scale_y
            camera_intrinsics[0, 2] *= scale_x
            camera_intrinsics[1, 2] *= scale_y
        h, w = depth_map.shape

        depth_m = (depth_map.astype(np.float32) / 1000.0
                   if np.median(depth_map[depth_map > 0]) > 100
                   else depth_map.astype(np.float32))

        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        lower_y  = int(h * 0.30)
        row_mask = np.zeros((h, w), dtype=bool)
        row_mask[lower_y:, :] = True

        valid_mask = ((depth_m >= MIN_DISTANCE_M) &
                      (depth_m <  MAX_DISTANCE_M) &
                      row_mask)
        if tag_masks is not None:
            valid_mask = valid_mask & (tag_masks == 0)
        valid_mask[int(h * 0.92):, :] = False   # exclude robot body edge

        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) < self.min_inliers:
            return self._cached_as_groundplane()

        x_cam = (x_coords[valid_mask] - cx) * depth_m[valid_mask] / fx
        y_cam = (y_coords[valid_mask] - cy) * depth_m[valid_mask] / fy
        z_cam = depth_m[valid_mask]
        points_3d = np.column_stack([x_cam, y_cam, z_cam])

        n_valid       = len(points_3d)
        subsample_idx = None
        if n_valid > MAX_RANSAC_PTS:
            subsample_idx = np.random.choice(n_valid, MAX_RANSAC_PTS, replace=False)
            points_3d     = points_3d[subsample_idx]

        best_normal, best_distance, best_inliers = None, None, 0
        best_inlier_mask = None

        for _ in range(self.max_iterations):
            if len(points_3d) < 3:
                break
            idx    = np.random.choice(len(points_3d), 3, replace=False)
            sample = points_3d[idx]
            v1, v2 = sample[1] - sample[0], sample[2] - sample[0]
            normal = np.cross(v1, v2)
            nl     = np.linalg.norm(normal)
            if nl < 1e-6:
                continue
            normal   /= nl
            distance  = -np.dot(normal, sample[0])
            dists     = np.abs(np.dot(points_3d, normal) + distance)
            imask     = dists < self.ransac_threshold
            cnt       = int(np.sum(imask))
            if cnt > best_inliers:
                best_inliers, best_normal = cnt, normal
                best_distance, best_inlier_mask = distance, imask

        if best_inliers < self.min_inliers or best_normal is None:
            return self._cached_as_groundplane()

        # SVD refinement on inliers
        inliers  = points_3d[best_inlier_mask]
        centroid = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal   = vh[-1]

        # FIX: camera-frame Y points DOWN; the ground-plane normal must point
        # UP (+Y). The expected_normal has +Y from sin(downward pitch).
        # Align to it — fixes the sign inversion that caused the floor to be
        # treated as an obstacle.
        if normal[1] > 0:
            normal = -normal
        plane_d    = -np.dot(normal, centroid)
        confidence = min(1.0, best_inliers / len(points_3d))

        if subsample_idx is not None:
            inlier_vi_pos = subsample_idx[best_inlier_mask]
        else:
            inlier_vi_pos = np.where(best_inlier_mask)[0]

        vi_y = valid_indices[0][inlier_vi_pos]
        vi_x = valid_indices[1][inlier_vi_pos]
        bounds = ((int(np.min(vi_x)), int(np.max(vi_x)),
                   int(np.min(vi_y)), int(np.max(vi_y)))
                  if len(vi_x) > 0 else (0, w, 0, h))

        self._plane_cache = (normal, plane_d)
        return GroundPlane(normal=normal, distance=plane_d,
                           confidence=confidence, inliers=best_inliers,
                           bounds=bounds)

    def _cached_as_groundplane(self) -> Optional[GroundPlane]:
        if self._plane_cache is None:
            return None
        n, d = self._plane_cache
        return GroundPlane(normal=n, distance=d,
                           confidence=0.5, inliers=0, bounds=(0, 0, 0, 0))


# ---------------------------------------------------------------------------
# ObstacleDetector  (depth-only)
# ---------------------------------------------------------------------------
class ObstacleDetector:
    """
    Detects obstacles as regions significantly above the ground plane.
    Returns a list of Obstacle dataclass objects AND internal state
    (height_map_mm, protrusion_mask) so RGBDepthFusion can fuse without
    recomputing.
    """

    DISTANCE_ZONES = [
        {'max_m': 1.0, 'height_thresh_mm': 120, 'min_area_px': 150,
         'merge_dist_px': 30, 'median_h_min': 100},
        {'max_m': 2.0, 'height_thresh_mm':  80, 'min_area_px': 100,
         'merge_dist_px': 50, 'median_h_min':  60},
        {'max_m': 2.5, 'height_thresh_mm':  60, 'min_area_px':  80,
         'merge_dist_px': 70, 'median_h_min':  40},
    ]
    STEREO_CONF_THRESH = 150

    def __init__(self,
                 min_distance: float = MIN_DISTANCE_M,
                 max_distance: float = MAX_DISTANCE_M):
        self.min_distance = min_distance
        self.max_distance = max_distance

        self._kernel5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
        self._kernel7  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,  7))
        self._kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def _get_zone(self, distance_m: float) -> dict:
        for zone in self.DISTANCE_ZONES:
            if distance_m <= zone['max_m']:
                return zone
        return self.DISTANCE_ZONES[-1]

    def _fill_mask_holes(self, mask: np.ndarray) -> np.ndarray:
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(padded, None, (0, 0), 255)
        bg = padded[1:-1, 1:-1]
        return cv2.bitwise_or(mask, cv2.bitwise_not(bg))

    def _depth_grow_mask(self, protrusion_mask: np.ndarray,
                          depth_m: np.ndarray,
                          depth_tol_m: float = 0.12) -> np.ndarray:
        valid_depths = depth_m[protrusion_mask > 0]
        if len(valid_depths) == 0:
            return protrusion_mask
        rep_depth = float(np.percentile(valid_depths, 25))
        depth_compatible = (
            (np.abs(depth_m - rep_depth) < depth_tol_m) &
            (depth_m >= self.min_distance) &
            (depth_m <= self.max_distance)
        ).astype(np.uint8)
        dilated = cv2.dilate(protrusion_mask, self._kernel11, iterations=1)
        grown   = cv2.bitwise_and(dilated, depth_compatible * 255)
        return cv2.bitwise_or(grown, protrusion_mask)

    def _merge_nearby_contours(self, contours: list, depth_m: np.ndarray,
                                h: int, w: int, merge_dist_px: int) -> list:
        if len(contours) <= 1:
            return contours

        rects = [cv2.boundingRect(c) for c in contours]

        def rect_dist(r1, r2):
            dx = max(0, max(r1[0], r2[0]) - min(r1[0]+r1[2], r2[0]+r2[2]))
            dy = max(0, max(r1[1], r2[1]) - min(r1[1]+r1[3], r2[1]+r2[3]))
            return (dx*dx + dy*dy) ** 0.5

        def centre_depth(r):
            cy_r = min(r[1] + r[3]//2, h-1)
            cx_r = min(r[0] + r[2]//2, w-1)
            return float(depth_m[cy_r, cx_r])

        depths = [centre_depth(r) for r in rects]
        parent = list(range(len(contours)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(contours)):
            for j in range(i+1, len(contours)):
                if (rect_dist(rects[i], rects[j]) < merge_dist_px and
                        abs(depths[i] - depths[j]) < 0.3):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        groups: dict = {}
        for i in range(len(contours)):
            groups.setdefault(find(i), []).append(i)

        merged = []
        for gidx in groups.values():
            gc = [contours[i] for i in gidx]
            if len(gc) == 1:
                merged.append(gc[0])
            else:
                tmp = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(tmp, gc, -1, 255, thickness=cv2.FILLED)
                tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, self._kernel15)
                new_cnts, _ = cv2.findContours(
                    tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if new_cnts:
                    merged.append(max(new_cnts, key=cv2.contourArea))
        return merged

    def _build_zoned_protrusion_mask(self, height_map_mm: np.ndarray,
                                      depth_m: np.ndarray,
                                      valid_mask: np.ndarray) -> np.ndarray:
        h, w       = height_map_mm.shape
        thresh_map = np.full((h, w), 999.0, dtype=np.float32)
        for zone in reversed(self.DISTANCE_ZONES):
            thresh_map[depth_m <= zone['max_m']] = zone['height_thresh_mm']
        out = np.zeros((h, w), dtype=np.uint8)
        out[valid_mask] = (height_map_mm[valid_mask] > thresh_map[valid_mask]
                           ).astype(np.uint8) * 255
        return out

    def detect_obstacles(self,
                          depth_map:         np.ndarray,
                          ground_plane:      GroundPlane,
                          camera_intrinsics: np.ndarray,
                          tag_masks:         Optional[np.ndarray] = None,
                          confidence_map:    Optional[np.ndarray] = None,
                          ) -> Tuple[List[Obstacle], np.ndarray, np.ndarray]:
        """
        Returns (obstacles, height_map_mm, protrusion_mask).
        height_map_mm and protrusion_mask are passed to RGBDepthFusion.fuse()
        so they are not recomputed there.
        """
        orig_h, orig_w = depth_map.shape
        if orig_h > 240 or orig_w > 320:
            depth_map = cv2.resize(depth_map, (320, 240),
                                   interpolation=cv2.INTER_NEAREST)
            if tag_masks is not None:
                tag_masks = cv2.resize(tag_masks, (320, 240),
                                       interpolation=cv2.INTER_NEAREST)
            if confidence_map is not None:
                confidence_map = cv2.resize(confidence_map, (320, 240),
                                            interpolation=cv2.INTER_NEAREST)
            scale_x = 320 / orig_w
            scale_y = 240 / orig_h
            camera_intrinsics = camera_intrinsics.copy()
            camera_intrinsics[0, 0] *= scale_x
            camera_intrinsics[1, 1] *= scale_y
            camera_intrinsics[0, 2] *= scale_x
            camera_intrinsics[1, 2] *= scale_y
        h, w = depth_map.shape

        depth_m = (depth_map.astype(np.float32) / 1000.0
                   if np.median(depth_map[depth_map > 0]) > 100
                   else depth_map.astype(np.float32))

        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        valid_mask = (depth_m >= self.min_distance) & (depth_m <= self.max_distance)
        if tag_masks is not None:
            valid_mask = valid_mask & (tag_masks == 0)
        empty_hmap = np.zeros((h, w), dtype=np.float32)
        if not np.any(valid_mask):
            return [], empty_hmap, np.zeros((h, w), dtype=np.uint8)

        y_idx, x_idx = np.where(valid_mask)
        z   = depth_m[y_idx, x_idx]
        pts = np.column_stack([
            (x_idx - cx) * z / fx,
            (y_idx - cy) * z / fy,
            z,
        ])

        n, d = ground_plane.normal, ground_plane.distance
        # FIX: np.maximum(0,…) instead of np.abs(…)
        # With an upward-pointing normal, pixels above the floor give positive
        # signed distance; the floor itself gives ≈0; sub-plane pixels give
        # negative values that must NOT be treated as obstacles.
        heights_mm    = np.maximum(0.0, pts @ n + d) * 1000.0
        height_map_mm = np.zeros((h, w), dtype=np.float32)
        height_map_mm[y_idx, x_idx] = heights_mm

        protrusion_mask = self._build_zoned_protrusion_mask(
            height_map_mm, depth_m, valid_mask)

        if confidence_map is not None:
            conf_mask       = (confidence_map >= self.STEREO_CONF_THRESH
                               ).astype(np.uint8) * 255
            protrusion_mask = cv2.bitwise_and(protrusion_mask, conf_mask)

        protrusion_mask = cv2.morphologyEx(protrusion_mask,
                                            cv2.MORPH_CLOSE, self._kernel7)
        protrusion_mask = self._fill_mask_holes(protrusion_mask)
        protrusion_mask = self._depth_grow_mask(protrusion_mask, depth_m)
        protrusion_mask = cv2.morphologyEx(protrusion_mask,
                                            cv2.MORPH_CLOSE, self._kernel7)

        contours, _ = cv2.findContours(
            protrusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = self._merge_nearby_contours(
            list(contours), depth_m, h, w, merge_dist_px=50)

        obstacles  = []
        cnt_mask   = np.zeros((h, w), dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            cx_box = x_b + w_b // 2
            cy_box = y_b + h_b // 2

            if cy_box >= h or cx_box >= w:
                continue
            z_val = depth_m[cy_box, cx_box]
            if z_val < self.min_distance or z_val > self.max_distance:
                continue

            zone = self._get_zone(z_val)
            if area < zone['min_area_px']:
                continue

            cnt_mask[y_b:y_b+h_b, x_b:x_b+w_b] = 0
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            contour_heights = height_map_mm[cnt_mask == 255]
            median_h = float(np.median(contour_heights)) if len(contour_heights) else 0.0
            cnt_mask[y_b:y_b+h_b, x_b:x_b+w_b] = 0

            if median_h < zone['median_h_min']:
                continue

            pos_x    = (cx_box - cx) * z_val / fx
            pos_y    = (cy_box - cy) * z_val / fy
            distance = float(np.sqrt(pos_x**2 + pos_y**2 + z_val**2))
            bearing  = float(np.arctan2(pos_x, z_val))
            severity = min(1.0, median_h / 300.0) * (1.0 / max(0.1, distance))

            comp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(comp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            cnts_out, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            contour_out = max(cnts_out, key=cv2.contourArea) if cnts_out else None

            obstacles.append(Obstacle(
                center=(cx_box, cy_box),
                position_3d=np.array([pos_x, pos_y, z_val]),
                size=(w_b, h_b),
                distance=distance,
                bearing=bearing,
                severity=severity,
                contour=contour_out,
                detection_source='depth',
            ))

        obstacles.sort(key=lambda o: o.distance)
        return obstacles, height_map_mm, protrusion_mask


# ---------------------------------------------------------------------------
# RGBDepthFusion  — FastSAM on a background thread, fused each depth frame
# ---------------------------------------------------------------------------
class RGBDepthFusion:
    """
    FastSAM runs asynchronously on a background thread so it never blocks
    the depth pipeline.  fuse() uses the most recent completed result
    (may be 1–3 frames stale — acceptable; segmentation masks change slowly).
    """

    def __init__(self,
                 model_path:         str   = "FastSAM-s.pt",
                 confidence:         float = 0.4,
                 iou_threshold:      float = 0.9,
                 min_mask_area_frac: float = 0.003,
                 max_mask_area_frac: float = 0.8):
        print("[FUSION] Loading FastSAM model...")
        self.model               = FastSAM(model_path)
        self.confidence          = confidence
        self.iou_threshold       = iou_threshold
        self.min_mask_area_frac  = min_mask_area_frac
        self.max_mask_area_frac  = max_mask_area_frac

        self.floor_hsv_mean: Optional[np.ndarray] = None
        self.floor_hsv_std:  Optional[np.ndarray] = None

        self._lock            = threading.Lock()
        self._latest_masks:   List[np.ndarray]       = []
        self._pending_frame:  Optional[np.ndarray]   = None
        self._thread          = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._floor_update_counter = 0
        print("[FUSION] FastSAM background thread started.")

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------
    def _worker(self):
        while True:
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame, self._pending_frame = self._pending_frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            masks = self._run_fastsam_sync(frame)
            with self._lock:
                self._latest_masks = masks

    def submit_frame(self, rgb_frame: np.ndarray):
        """Queue a frame for FastSAM inference (drops stale pending frame)."""
        with self._lock:
            self._pending_frame = rgb_frame.copy()

    def get_latest_masks(self) -> List[np.ndarray]:
        with self._lock:
            return list(self._latest_masks)

    # ------------------------------------------------------------------
    # FastSAM inference (runs on background thread)
    # ------------------------------------------------------------------
    def _run_fastsam_sync(self, rgb_frame: np.ndarray) -> List[np.ndarray]:
        h, w       = rgb_frame.shape[:2]
        frame_area = h * w

        results = self.model(
            rgb_frame,
            device='cpu',
            retina_masks=False,
            imgsz=FASTSAM_IMGSZ,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )

        if not results or results[0].masks is None:
            return []

        masks = []
        for mt in results[0].masks.data:
            mask = mt.cpu().numpy().astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            af = np.sum(mask) / frame_area
            if self.min_mask_area_frac <= af <= self.max_mask_area_frac:
                masks.append(mask)

        # Deduplicate by IoU
        if len(masks) > 1:
            masks.sort(key=lambda m: int(np.sum(m)), reverse=True)
            flat  = np.stack([m.ravel().astype(np.float32) for m in masks])
            areas = flat.sum(axis=1)
            keep: List[np.ndarray]       = []
            kept_flat: List[tuple]       = []
            for i, f in enumerate(flat):
                if any(
                    float(np.dot(f, kf)) /
                    max(1.0, float(areas[i] + areas[ki] - np.dot(f, kf))) > 0.8
                    for ki, kf in kept_flat
                ):
                    continue
                keep.append(masks[i])
                kept_flat.append((i, f))
            masks = keep

        return masks

    # ------------------------------------------------------------------
    # Floor HSV model
    # ------------------------------------------------------------------
    def _update_floor_model(self, rgb_bgr: np.ndarray, depth_m: np.ndarray,
                             ground_normal: np.ndarray, plane_d: float,
                             K: np.ndarray):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        y_idx, x_idx   = np.where((depth_m > 0.3) & (depth_m < 2.5))
        if len(y_idx) == 0:
            return
        z   = depth_m[y_idx, x_idx]
        pts = np.column_stack([
            (x_idx - cx) * z / fx,
            (y_idx - cy) * z / fy,
            z,
        ])
        # FIX: np.maximum(0,…) — floor pixels sit just above the plane (small
        # positive height). abs() previously included sub-plane pixels and
        # polluted the HSV floor model with ceiling/distant background colours.
        heights    = np.maximum(0.0, pts @ ground_normal + plane_d) * 1000.0
        floor_mask = heights < 40
        fy_idx, fx_idx = y_idx[floor_mask], x_idx[floor_mask]
        if len(fy_idx) < 50:
            return
        hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        fp  = hsv[fy_idx, fx_idx]
        self.floor_hsv_mean = np.mean(fp, axis=0)
        self.floor_hsv_std  = np.std(fp, axis=0) + 1e-6

    def _is_floor_segment(self, mask: np.ndarray, hsv_frame: np.ndarray) -> bool:
        if self.floor_hsv_mean is None:
            return False
        px = hsv_frame[mask > 0]
        if len(px) == 0:
            return False
        z = np.abs(px.mean(axis=0) - self.floor_hsv_mean) / self.floor_hsv_std
        return float(z[0]) < 2.0 and float(z[1]) < 2.0

    def _get_mask_depth_stats(self, mask: np.ndarray, depth_m: np.ndarray,
                               min_dist: float, max_dist: float):
        interior = depth_m[mask > 0]
        valid    = interior[(interior > min_dist) & (interior < max_dist)]
        if len(valid) > 20:
            return {'median_depth_m': float(np.median(valid)),
                    'depth_coverage': len(valid) / max(1, len(interior)),
                    'source': 'interior'}
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask, kernel, iterations=2)
        ring    = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))
        edge    = depth_m[ring > 0]
        valid_e = edge[(edge > min_dist) & (edge < max_dist)]
        if len(valid_e) > 10:
            return {'median_depth_m': float(np.median(valid_e)),
                    'depth_coverage': 0.0,
                    'source': 'edge_ring'}
        return None

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        i = np.logical_and(a, b).sum()
        u = np.logical_or(a, b).sum()
        return float(i) / max(1, float(u))

    @staticmethod
    def _contour_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        cs, _ = cv2.findContours(mask.astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(cs, key=cv2.contourArea) if cs else None

    # ------------------------------------------------------------------
    # fuse() — takes depth Obstacle list, returns fused Obstacle list
    # ------------------------------------------------------------------
    def fuse(self,
             rgb_bgr:         np.ndarray,
             depth_m:         np.ndarray,
             depth_obstacles: List[Obstacle],
             ground_plane:    GroundPlane,
             K:               np.ndarray,
             height_map_mm:   np.ndarray,
             min_dist:        float = MIN_DISTANCE_M,
             max_dist:        float = MAX_DISTANCE_M,
             ) -> List[Obstacle]:
        """
        Fuse FastSAM RGB masks with the depth-derived obstacle list.
        Returns a list of Obstacle dataclass objects (same type as the input)
        annotated with detection_source 'depth', 'rgb', or 'depth+rgb'.
        """
        h, w           = depth_m.shape      # depth is already downsampled (e.g. 320×240)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # SAM masks are produced at the resolution of the submitted RGB frame
        # (640×400), while depth_m has been downsampled to 320×240.  Resize
        # rgb_bgr to depth resolution once here so that:
        #   • _update_floor_model indexes rgb_bgr with depth-space (y, x) coords
        #   • hsv_frame and every SAM mask share the same (h, w)
        if rgb_bgr.shape[0] != h or rgb_bgr.shape[1] != w:
            rgb_bgr = cv2.resize(rgb_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        self._floor_update_counter += 1
        if self._floor_update_counter % FLOOR_UPDATE_INTERVAL == 0:
            self._update_floor_model(rgb_bgr, depth_m,
                                     ground_plane.normal, ground_plane.distance, K)

        hsv_frame = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Pre-rasterise depth obstacle contours once for IoU matching
        d_masks = []
        for d_obs in depth_obstacles:
            dm = np.zeros((h, w), dtype=np.uint8)
            if d_obs.contour is not None:
                cv2.drawContours(dm, [d_obs.contour], -1, 255,
                                 thickness=cv2.FILLED)
            d_masks.append(dm)

        # Resize SAM masks from RGB resolution down to depth resolution
        raw_masks = self.get_latest_masks()
        rgb_masks = []
        for m in raw_masks:
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            rgb_masks.append(m)

        depth_matched = [False] * len(depth_obstacles)
        fused: List[Obstacle] = []

        for mask in rgb_masks:
            if self._is_floor_segment(mask, hsv_frame):
                continue

            ds = self._get_mask_depth_stats(mask, depth_m, min_dist, max_dist)
            if ds is None:
                continue

            mask_depth = ds['median_depth_m']
            contour    = self._contour_from_mask(mask)
            if contour is None:
                continue

            x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
            cx_box = x_b + w_b // 2
            cy_box = y_b + h_b // 2
            pos_x  = (cx_box - cx) * mask_depth / fx
            pos_y  = (cy_box - cy) * mask_depth / fy
            dist   = float(np.sqrt(pos_x**2 + pos_y**2 + mask_depth**2))

            if dist < min_dist or dist > max_dist:
                continue

            # Match against depth obstacles (early exit on strong IoU)
            matched_idx, best_iou = None, 0.15
            for di, dm in enumerate(d_masks):
                iou = self._mask_iou(mask, dm)
                if iou > best_iou:
                    best_iou, matched_idx = iou, di
                    if best_iou > 0.5:
                        break

            # Compute median height for this RGB mask region
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [contour], -1, 255, thickness=cv2.FILLED)
            ch = height_map_mm[cnt_mask == 255]
            median_h = float(np.median(ch)) if len(ch) else 0.0

            bearing  = float(np.arctan2(pos_x, mask_depth))
            severity = min(1.0, median_h / 300.0) * (1.0 / max(0.1, dist))

            if matched_idx is not None:
                depth_matched[matched_idx] = True
                d = depth_obstacles[matched_idx]
                fused.append(Obstacle(
                    center=d.center,
                    position_3d=d.position_3d,
                    size=d.size,
                    distance=d.distance,
                    bearing=d.bearing,
                    severity=max(d.severity, severity),
                    contour=contour,
                    detection_source='depth+rgb',
                ))
            else:
                if median_h < 30 and ds['source'] == 'edge_ring':
                    continue
                fused.append(Obstacle(
                    center=(cx_box, cy_box),
                    position_3d=np.array([pos_x, pos_y, mask_depth]),
                    size=(w_b, h_b),
                    distance=dist,
                    bearing=bearing,
                    severity=severity,
                    contour=contour,
                    detection_source='rgb',
                ))

        # Carry through unmatched depth-only obstacles
        for di, d_obs in enumerate(depth_obstacles):
            if not depth_matched[di]:
                fused.append(Obstacle(
                    center=d_obs.center,
                    position_3d=d_obs.position_3d,
                    size=d_obs.size,
                    distance=d_obs.distance,
                    bearing=d_obs.bearing,
                    severity=d_obs.severity,
                    contour=d_obs.contour,
                    detection_source='depth',
                ))

        fused.sort(key=lambda o: o.distance)
        return fused


# ---------------------------------------------------------------------------
# PathPlanner
# ---------------------------------------------------------------------------
class PathPlanner:
    """
    Reactive Vector Field Path Planner (Vectorized).
    """
    def __init__(self,
                 robot_width:   float = 0.5,
                 min_clearance: float = 0.2,
                 max_lookahead: float = 3.0):
        self.robot_width   = robot_width
        self.min_clearance = min_clearance
        self.max_lookahead = max_lookahead
        self.thetas        = np.deg2rad(np.arange(-40, 41, 1))
        self.w_target      = 1.0
        self.w_obs         = 5.0
        self.margin_rad    = np.deg2rad(5)

    def plan_path_to_tag(self, tag_detection, obstacles: list,
                          ground_plane, camera_intrinsics: np.ndarray,
                          image_shape: tuple) -> Optional[list]:
        tag_pos      = tag_detection.pose[:3, 3]
        tag_x, tag_z = tag_pos[0], tag_pos[2]
        target_dist  = np.hypot(tag_x, tag_z)
        if target_dist < 0.1:
            return None

        target_angle = np.arctan2(tag_x, tag_z)
        fx           = camera_intrinsics[0, 0]

        obs_blockers = []
        for obs in obstacles:
            obs_x, obs_z = obs.position_3d[0], obs.position_3d[2]
            d = np.hypot(obs_x, obs_z)
            if d < 0.1 or obs_z < 0:
                continue
            obs_width_m = (obs.size[0] * obs_z) / fx
            r_safe      = (obs_width_m / 2.0) + (self.robot_width / 2.0) + self.min_clearance
            obs_angle   = np.arctan2(obs_x, obs_z)
            if d <= r_safe:
                obs_blockers.append((obs_angle, np.pi, d))
            else:
                delta_theta = np.arcsin(np.clip(r_safe / d, -1.0, 1.0))
                obs_blockers.append((obs_angle, delta_theta, d))

        diff_target = self._wrap_angle(self.thetas - target_angle)
        cost_target = 1.0 - np.cos(diff_target)

        cost_obs = np.zeros_like(self.thetas)
        for obs_angle, delta_theta, d in obs_blockers:
            diff_obs     = self._wrap_angle(self.thetas - obs_angle)
            blocked_zone = delta_theta + self.margin_rad
            penetration  = np.maximum(0.0, 1.0 - (np.abs(diff_obs) / blocked_zone))
            cost_obs    += penetration * (1.0 / max(0.5, d))

        total_cost = (self.w_target * cost_target) + (self.w_obs * cost_obs)
        best_idx   = np.argmin(total_cost)
        best_angle = self.thetas[best_idx]

        lookahead = min(self.max_lookahead, target_dist)
        wp_x      = lookahead * np.sin(best_angle)
        wp_z      = lookahead * np.cos(best_angle)
        return [PathSegment(
            start=np.array([0.0, 0.0, 0.0]),
            end=np.array([wp_x, 0.0, wp_z]),
            width=self.robot_width,
            clearance=self.min_clearance,
            cost=float(total_cost[best_idx]),
        )]

    @staticmethod
    def _wrap_angle(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# GroundAndObstaclePipeline
# ---------------------------------------------------------------------------
class GroundAndObstaclePipeline:
    """
    Complete pipeline: ground detection → depth obstacle detection →
    FastSAM RGB fusion → path planning.
    """

    def __init__(self, robot_width: float = 0.5,
                 fastsam_model: str = "FastSAM-s.pt"):
        self.ground_detector   = GroundPlaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.fusion            = RGBDepthFusion(model_path=fastsam_model)
        self.path_planner      = PathPlanner(robot_width=robot_width)
        self.camera_intrinsics: Optional[np.ndarray] = None

        self._frame_count = 0

    def set_camera_intrinsics(self, fx: float, fy: float,
                               cx: float, cy: float):
        self.camera_intrinsics = np.array([[fx, 0, cx],
                                            [0, fy, cy],
                                            [0, 0,  1]], dtype=np.float32)

    def create_tag_mask(self, image_shape: tuple,
                        tag_detections: list) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for tag in tag_detections:
            if hasattr(tag, 'corners') and tag.corners is not None:
                pts = np.array(tag.corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
        if np.any(mask > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask   = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def process_frame(self,
                       depth_map:      np.ndarray,
                       rgb_bgr:        np.ndarray,
                       tag_detections: list,
                       image_shape:    tuple,
                       confidence_map: Optional[np.ndarray] = None) -> dict:
        """
        Process a single frame for ground and obstacle perception.

        Args:
            depth_map:       Raw depth from OAK-D (mm, uint16).
            rgb_bgr:         BGR frame from the camera (used by FastSAM fusion).
            tag_detections:  List of AprilTagDetection objects.
            image_shape:     (height, width) of the RGB frame.
            confidence_map:  Optional stereo confidence map (0–255).

        Returns dict with keys: 'ground_plane', 'obstacles', 'tag_mask'.
        """
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not set — call set_camera_intrinsics() first.")

        self._frame_count += 1

        tag_mask     = self.create_tag_mask(image_shape, tag_detections)
        ground_plane = self.ground_detector.detect_ground_plane(
            depth_map, self.camera_intrinsics, tag_mask)

        obstacles: List[Obstacle] = []

        if ground_plane is not None:
            depth_obstacles, height_map_mm, _ = self.obstacle_detector.detect_obstacles(
                depth_map, ground_plane, self.camera_intrinsics,
                tag_mask, confidence_map)

            # Submit frame to FastSAM background thread every FASTSAM_INTERVAL frames
            if self._frame_count % FASTSAM_INTERVAL == 0:
                self.fusion.submit_frame(rgb_bgr)

            # Fuse latest SAM masks (non-blocking — uses most recent background result)
            depth_m = (depth_map.astype(np.float32) / 1000.0
                       if np.median(depth_map[depth_map > 0]) > 100
                       else depth_map.astype(np.float32))
            # Downsample depth_m to match height_map_mm dimensions if needed
            if depth_m.shape != height_map_mm.shape:
                depth_m = cv2.resize(depth_m, (height_map_mm.shape[1],
                                               height_map_mm.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

            obstacles = self.fusion.fuse(
                rgb_bgr, depth_m, depth_obstacles,
                ground_plane, self.camera_intrinsics, height_map_mm)

        return {
            'ground_plane': ground_plane,
            'obstacles':    obstacles,
            'tag_mask':     tag_mask,
        }


if __name__ == "__main__":
    print("Ground plane, obstacle detection, and FastSAM fusion module loaded.")
    print("Use with apriltag_detection.py and main_navigation.py")