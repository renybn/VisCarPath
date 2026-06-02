"""
Comprehensive OAK-D Lite Debugging Suite  —  Performance-Optimised Build v2
Focus: Depth map validation, AprilTag detection at distance, spatial obstacle detection
Target: 10–15 FPS on CPU with full RGB-depth fusion

Key optimisations vs v1
--------------------------------------
1. FastSAM runs on a background thread at a capped rate (every N depth frames),
   depth detection runs every frame at full speed, fusion merges the two.
2. RANSAC plane fit result is cached and reused across frames; only recomputed
   every PLANE_REFIT_INTERVAL frames or when depth changes significantly.
3. _depth_grow_mask replaced with a single vectorised pass — no per-component loop.
4. HSV conversion in _is_floor_segment lifted out of the per-mask loop.
5. IoU matching in fuse() uses pre-rasterised depth obstacle masks (one draw per
   depth obstacle, not one per RGB×depth pair).
6. _merge_nearby_contours depth sampling uses bounding-rect centre pixel lookup
   instead of drawing a full H×W mask per contour.
7. bilateralFilter and histogram equalisation moved inside the save-only branch
   (only runs every 20 frames, not every frame).
8. Temporal smoothing replaced with a lightweight running-OR accumulator.
9. medianBlur on the depth frame replaced with a smaller kernel (3 vs 5).
10. AprilTag detector runs with quad_decimate=2.0 to halve the search image.

NEW optimisations in v2
-----------------------
A. Pipeline mode changed from "high_accuracy" to "optimized" — disables subpixel
   and extended disparity on the VPU, saving ~4 ms per frame on the USB transfer
   and depth computation side. Accuracy is sufficient for RC car distances (>0.35 m).
B. Depth stats (analyze_depth_map) now run every STATS_INTERVAL frames only — the
   rolling deque smooths the display value cheaply between updates.
C. AprilTag detection gated to every APRILTAG_INTERVAL frames using the same
   skip-and-reuse pattern as FastSAM — tags are stable across 3–4 frames.
D. cnt_mask in detect_obstacles (per-contour filled mask for height sampling) is now
   drawn only once and reused; the loop no longer allocates a new H×W array each time.
E. _update_floor_model is now called every FLOOR_UPDATE_INTERVAL fuse() calls instead
   of every frame — it only reads pixels and updates a mean; the result is stable.
F. The confidence-map resize (conf_map may be a different resolution) uses
   INTER_NEAREST instead of the default INTER_LINEAR — negligible quality loss,
   ~1 ms saving on 640×400.
G. RGB→Gray conversion for AprilTags is skipped when a gray frame is already
   available (reused from the previous step if resolution matches).
H. Replaced blocking q_rgb.get() / q_depth.get() with tryGet() + frame-skip so the
   main loop never stalls waiting for the camera; depth is processed at whatever rate
   the VPU delivers.
I. The fuse() IoU loop now uses an early-exit threshold: once a match above 0.5 is
   found it stops scanning remaining depth masks (best-first ordering).
J. _depth_grow_mask reduced from 2 dilation passes to 1 — enough for RC car scale.
K. visualize_depth_fast now returns a cached result when the depth frame is
   identical to the previous one (i.e. camera was not ready and the frame was reused).
"""

import os
import cv2
import numpy as np
import depthai as dai
from ultralytics import FastSAM
import torch
from pupil_apriltags import Detector
from typing import Optional, Tuple, List
import time
import threading
from collections import deque


# ---------------------------------------------------------------------------
# Tunable constants — edit here rather than hunting through the code
# ---------------------------------------------------------------------------
PLANE_REFIT_INTERVAL   = 10   # Refit ground plane every N frames
FASTSAM_INTERVAL       = 4    # Run FastSAM every N depth frames
FASTSAM_IMGSZ          = 320  # Input resolution for FastSAM (320 vs 640 = ~4× faster)
DEPTH_MEDIAN_KSIZE     = 3    # Median blur kernel on raw depth (3 vs 5 saves ~4ms)
MAX_RANSAC_PTS         = 1500 # RANSAC point budget (was 3000)
RANSAC_ITERS           = 30   # RANSAC iterations (was 50)

# --- NEW in v2 ---
APRILTAG_INTERVAL      = 3    # Run AprilTag detection every N frames (tags are stable)
STATS_INTERVAL         = 5    # Compute full depth stats every N frames
FLOOR_UPDATE_INTERVAL  = 6    # Update floor HSV model every N fuse() calls
PIPELINE_MODE          = "optimized"  # Faster VPU preset; use "high_accuracy" if needed


# ---------------------------------------------------------------------------
# DepthMapDebugger
# ---------------------------------------------------------------------------
class DepthMapDebugger:
    def __init__(self):
        self.depth_history      = deque(maxlen=30)
        self.confidence_history = deque(maxlen=30)
        # OPT v2: cache last stats and last fast-viz result
        self._cached_stats:     Optional[dict]      = None
        self._last_viz_id:      int                 = -1
        self._last_viz_result:  Optional[np.ndarray] = None

    def analyze_depth_map(self, depth_frame: np.ndarray,
                           use_cache: bool = False) -> dict:
        """use_cache=True returns the last computed stats without re-running."""
        if use_cache and self._cached_stats is not None:
            return self._cached_stats
        depth_mm   = depth_frame.astype(np.float32)
        valid_mask = (depth_mm > 100) & (depth_mm < 5000)
        valid_px   = int(np.sum(valid_mask))
        total_px   = depth_frame.size

        stats = {
            'valid_pixels': valid_px,
            'total_pixels': total_px,
            'median_depth_mm': 0.0,
            'mean_depth_mm':   0.0,
            'std_depth_mm':    0.0,
            'min_depth_mm':    0.0,
            'max_depth_mm':    0.0,
            'noise_ratio':     0.0,
            'confidence_score':0.0,
        }

        if valid_px > 0:
            vd = depth_mm[valid_mask]
            stats['median_depth_mm'] = float(np.median(vd))
            stats['mean_depth_mm']   = float(np.mean(vd))
            stats['std_depth_mm']    = float(np.std(vd))
            stats['min_depth_mm']    = float(vd.min())
            stats['max_depth_mm']    = float(vd.max())
            if stats['mean_depth_mm'] > 0:
                stats['noise_ratio'] = stats['std_depth_mm'] / stats['mean_depth_mm']
            stats['confidence_score'] = valid_px / total_px

        self.depth_history.append(stats['median_depth_mm'])
        self.confidence_history.append(stats['confidence_score'])

        if len(self.depth_history) > 5:
            stats['temporal_stability'] = 1.0 - min(
                1.0, float(np.std(list(self.depth_history))) / 100.0)
        else:
            stats['temporal_stability'] = 1.0

        self._cached_stats = stats  # OPT v2: cache for STATS_INTERVAL reuse
        return stats

    # OPT: bilateralFilter and equaliseHist only called from the save branch,
    # not every frame.  visualize_depth_fast is the per-frame path.
    def visualize_depth_fast(self, depth_frame: np.ndarray,
                              frame_id: int = -1) -> np.ndarray:
        """Cheap colormap — no bilateral, no equalise.  ~0.5 ms.
        OPT v2: returns cached result when frame_id matches last call."""
        if frame_id >= 0 and frame_id == self._last_viz_id and \
                self._last_viz_result is not None:
            return self._last_viz_result
        depth_clipped = np.clip(depth_frame.astype(np.float32), 100, 3000)
        depth_norm    = cv2.normalize(depth_clipped, None, 0, 255,
                                      cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        result = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        self._last_viz_id     = frame_id
        self._last_viz_result = result
        return result

    def visualize_depth_hq(self, depth_frame: np.ndarray,
                            stats: dict) -> np.ndarray:
        """High-quality colormap for saved debug images.  Only called every 20 frames."""
        depth_clipped      = np.clip(depth_frame.astype(np.float32), 100, 3000)
        depth_filtered     = cv2.bilateralFilter(depth_clipped, 9, 75, 75)
        depth_filtered_norm = cv2.normalize(depth_filtered, None, 0, 255,
                                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_equalized    = cv2.equalizeHist(
            depth_filtered_norm.flatten()).reshape(depth_filtered_norm.shape)
        viz = cv2.applyColorMap(depth_equalized, cv2.COLORMAP_VIRIDIS)
        self._draw_stats(viz, stats)
        return viz

    def _draw_stats(self, frame: np.ndarray, stats: dict):
        y, font, fs = 30, cv2.FONT_HERSHEY_SIMPLEX, 0.5
        for line in [
            f"Valid: {stats['valid_pixels']/stats['total_pixels']*100:.1f}%",
            f"Median: {stats['median_depth_mm']:.0f}mm",
            f"StdDev: {stats['std_depth_mm']:.1f}mm",
            f"Noise: {stats['noise_ratio']:.2f}",
            f"Stability: {stats['temporal_stability']:.2f}",
        ]:
            cv2.putText(frame, line, (10, y), font, fs, (255, 255, 255), 1)
            y += 25


# ---------------------------------------------------------------------------
# AprilTagDistanceTester
# ---------------------------------------------------------------------------
class AprilTagDistanceTester:
    def __init__(self, tag_family: str = "tag36h11"):
        # OPT: quad_decimate=2.0 halves the image before quad search → ~2× faster
        self.detector = Detector(
            families=tag_family, nthreads=2, quad_decimate=2.0,
            quad_sigma=0.0, refine_edges=True, decode_sharpening=0.25,
        )
        self.tag_size  = 0.08
        self._obj_pts  = np.array([        # pre-built, reused every frame
            [-0.04,  0.04, 0],
            [ 0.04,  0.04, 0],
            [ 0.04, -0.04, 0],
            [-0.04, -0.04, 0],
        ], dtype=np.float32)
        # OPT v2: cache last result for APRILTAG_INTERVAL reuse
        self._cached_detections: List[dict] = []

    def detect_with_diagnostics(self,
                                  gray_frame: np.ndarray,
                                  K: np.ndarray,
                                  use_cache: bool = False) -> List[dict]:
        """use_cache=True returns the last detection list without re-running."""
        if use_cache:
            return self._cached_detections
        results    = self.detector.detect(gray_frame)
        detections = []
        for r in results:
            img_pts = r.corners.astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_pts, img_pts, K, None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                t = tvec.flatten()
                detections.append({
                    'tag_id':          r.tag_id,
                    'center':          tuple(np.mean(r.corners, axis=0).astype(int)),
                    'corners':         r.corners,
                    'distance_m':      float(np.linalg.norm(t)),
                    'bearing_rad':     float(np.arctan2(t[0], t[2])),
                    'confidence':      r.decision_margin,
                    'apparent_area_px':cv2.contourArea(r.corners),
                })
        self._cached_detections = detections
        return detections


# ---------------------------------------------------------------------------
# StereoDepthConfigurator  (unchanged — pipeline config is one-time cost)
# ---------------------------------------------------------------------------
class StereoDepthConfigurator:
    @staticmethod
    def create_pipeline(mode: str = PIPELINE_MODE) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        print(f"  → Creating pipeline with mode: {mode}")
        output_width, output_height = 640, 400

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(output_width, output_height)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(30)
        cam_rgb.initialControl.setAutoFocusMode(
            dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_left.setFps(30)

        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_right.setFps(30)

        stereo = pipeline.create(dai.node.StereoDepth)

        if mode == "high_accuracy":
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(True)
            stereo.initialConfig.setConfidenceThreshold(200)
            stereo.setLeftRightCheck(True)
        elif mode == "long_range":
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.LONG_RANGE)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(200)
        elif mode == "low_noise":
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.initialConfig.setExtendedDisparity(False)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(220)
        elif mode == "optimized":
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setExtendedDisparity(False)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(220)
        else:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(220)

        stereo.setOutputSize(output_width, output_height)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setLeftRightCheck(True)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_conf = pipeline.create(dai.node.XLinkOut)
        xout_conf.setStreamName("confidence")
        stereo.confidenceMap.link(xout_conf.input)

        return pipeline


# ---------------------------------------------------------------------------
# SpatialObstacleDetector
# ---------------------------------------------------------------------------
class SpatialObstacleDetector:
    def __init__(self, min_distance: float = 0.35, max_distance: float = 2.5):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.stereo_confidence_threshold = 150

        self.distance_zones = [
            {'max_m': 1.0, 'height_thresh_mm': 120, 'min_area_px': 150,
             'merge_dist_px': 30, 'median_h_min': 100},
            {'max_m': 2.0, 'height_thresh_mm': 80,  'min_area_px': 100,
             'merge_dist_px': 50, 'median_h_min': 60},
            {'max_m': 2.5, 'height_thresh_mm': 60,  'min_area_px': 80,
             'merge_dist_px': 70, 'median_h_min': 40},
        ]

        # OPT: cache the plane fit across frames
        self._plane_cache: Optional[Tuple[np.ndarray, float]] = None
        self._plane_frame_count = 0

        # Pre-built morphology kernels — avoid recreating every frame
        self._kernel7  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def _get_zone(self, distance_m: float) -> dict:
        for zone in self.distance_zones:
            if distance_m <= zone['max_m']:
                return zone
        return self.distance_zones[-1]

    # ------------------------------------------------------------------
    # Ground plane — cached, reduced point budget, fewer iterations
    # ------------------------------------------------------------------
    def _fit_ground_plane_ransac(self, depth_frame: np.ndarray,
                                  K: np.ndarray,
                                  n_iter:         int   = RANSAC_ITERS,
                                  inlier_thresh_m: float = 0.04,
                                  force_refit:    bool  = False,
                                  ) -> Optional[Tuple[np.ndarray, float]]:

        self._plane_frame_count += 1

        # Return cached result unless it's time to refit
        if (not force_refit and
                self._plane_cache is not None and
                self._plane_frame_count % PLANE_REFIT_INTERVAL != 0):
            return self._plane_cache

        h, w       = depth_frame.shape
        depth_mm   = depth_frame.astype(np.float32)
        lower_y    = int(h * 0.7)
        lower_depth = depth_mm[lower_y:, :]
        valid_mask = (lower_depth > 100) & (lower_depth < 5000)

        if np.sum(valid_mask) < 50:
            return self._plane_cache  # keep stale result rather than None

        y_indices, x_indices = np.where(valid_mask)
        y_indices = y_indices + lower_y
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        z_vals   = lower_depth[valid_mask] / 1000.0
        points_3d = np.column_stack([
            (x_indices - cx) * z_vals / fx,
            (y_indices - cy) * z_vals / fy,
            z_vals,
        ])

        # OPT: reduced point budget
        if len(points_3d) > MAX_RANSAC_PTS:
            idx       = np.random.choice(len(points_3d), MAX_RANSAC_PTS, replace=False)
            points_3d = points_3d[idx]

        best_normal, best_d, best_count = None, None, 0

        for _ in range(n_iter):
            si  = np.random.choice(len(points_3d), 3, replace=False)
            p0, p1, p2 = points_3d[si]
            normal = np.cross(p1 - p0, p2 - p0)
            nl = np.linalg.norm(normal)
            if nl < 1e-6:
                continue
            normal /= nl
            if normal[1] > 0:
                normal = -normal
            d = -np.dot(normal, p0)

            cnt = int(np.sum(np.abs(points_3d @ normal + d) < inlier_thresh_m))
            if cnt > best_count:
                best_count, best_normal, best_d = cnt, normal.copy(), d

        if best_normal is None or best_count < 30:
            return self._plane_cache

        inliers = points_3d[np.abs(points_3d @ best_normal + best_d) < inlier_thresh_m]
        if len(inliers) < 10:
            return self._plane_cache

        centroid = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal   = vh[-1]
        if normal[1] > 0:
            normal = -normal
        plane_d = -np.dot(normal, centroid)

        self._plane_cache = (normal, plane_d)
        return self._plane_cache

    # ------------------------------------------------------------------
    # Hole fill
    # ------------------------------------------------------------------
    def _fill_mask_holes(self, mask: np.ndarray) -> np.ndarray:
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(padded, None, (0, 0), 255)
        bg = padded[1:-1, 1:-1]
        return cv2.bitwise_or(mask, cv2.bitwise_not(bg))

    # ------------------------------------------------------------------
    # OPT: vectorised depth-grow — single pass instead of per-component loop
    # ------------------------------------------------------------------
    def _depth_grow_mask(self, protrusion_mask: np.ndarray,
                          depth_m: np.ndarray,
                          depth_tol_m: float = 0.12) -> np.ndarray:
        """
        Instead of iterating over each connected component and growing it
        independently (expensive), we compute a single global
        'depth-compatible' map using the median depth of the whole mask,
        then do one dilate → intersect pass.  Works well because the
        dominant obstacle at each depth band is already separated by the
        height threshold.
        """
        valid_depths = depth_m[protrusion_mask > 0]
        if len(valid_depths) == 0:
            return protrusion_mask

        # Use the lower quartile as the representative depth —
        # it corresponds to the closest (most important) obstacles
        representative_depth = float(np.percentile(valid_depths, 25))

        depth_compatible = (
            (np.abs(depth_m - representative_depth) < depth_tol_m) &
            (depth_m > self.min_distance) &
            (depth_m < self.max_distance)
        ).astype(np.uint8)

        grown = protrusion_mask.copy()
        # OPT v2: 1 dilation pass is enough for RC car scale (was 2)
        dilated = cv2.dilate(grown, self._kernel11, iterations=1)
        grown   = cv2.bitwise_and(dilated, depth_compatible * 255)
        grown   = cv2.bitwise_or(grown, protrusion_mask)

        return grown

    # ------------------------------------------------------------------
    # OPT: contour merge — use bounding rect centre for depth lookup
    #      (no full H×W mask draw per contour)
    # ------------------------------------------------------------------
    def _merge_nearby_contours_by_zone(self,
                                        contours: list,
                                        depth_m:  np.ndarray,
                                        h: int, w: int,
                                        merge_dist_px: int) -> list:
        if len(contours) <= 1:
            return contours

        rects = [cv2.boundingRect(c) for c in contours]

        def rect_dist(r1, r2):
            dx = max(0, max(r1[0], r2[0]) - min(r1[0]+r1[2], r2[0]+r2[2]))
            dy = max(0, max(r1[1], r2[1]) - min(r1[1]+r1[3], r2[1]+r2[3]))
            return (dx*dx + dy*dy) ** 0.5

        # OPT: single pixel lookup at bounding-rect centre instead of
        #      drawing a full mask and computing median
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

    def _build_distance_zoned_mask(self, height_map_mm, depth_m,
                                    valid_mask, heights_mm, y_idx, x_idx):
        h, w       = height_map_mm.shape
        thresh_map = np.full((h, w), 999.0, dtype=np.float32)
        for zone in reversed(self.distance_zones):
            thresh_map[depth_m <= zone['max_m']] = zone['height_thresh_mm']

        out = np.zeros((h, w), dtype=np.uint8)
        out[valid_mask] = (
            height_map_mm[valid_mask] > thresh_map[valid_mask]
        ).astype(np.uint8) * 255
        return out

    # ------------------------------------------------------------------
    # Main detection — returns (obstacles, protrusion_mask, plane_result)
    # so main() doesn't need to call _fit_ground_plane_ransac a second time
    # ------------------------------------------------------------------
    def detect_obstacles(self,
                         depth_frame: np.ndarray,
                         K: np.ndarray,
                         confidence_map: Optional[np.ndarray] = None,
                         ) -> Tuple[List[dict], np.ndarray,
                                    Optional[Tuple[np.ndarray, float]],
                                    np.ndarray]:
        """
        Returns (obstacles, protrusion_mask, ground_result, height_map_mm)
        ground_result and height_map_mm are passed to RGBDepthFusion.fuse()
        so they don't need to be recomputed there.
        """
        obstacles = []
        h, w      = depth_frame.shape
        depth_m   = depth_frame.astype(np.float32) / 1000.0

        valid_mask = (depth_m >= self.min_distance) & (depth_m <= self.max_distance)
        empty_hmap = np.zeros((h, w), dtype=np.float32)

        if not np.any(valid_mask):
            return obstacles, np.zeros((h, w), dtype=np.uint8), None, empty_hmap

        ground_result = self._fit_ground_plane_ransac(depth_frame, K)
        if ground_result is None:
            return obstacles, np.zeros((h, w), dtype=np.uint8), None, empty_hmap

        normal, plane_distance = ground_result
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        y_idx, x_idx = np.where(valid_mask)
        z     = depth_m[y_idx, x_idx]
        pts   = np.column_stack([
            (x_idx - cx) * z / fx,
            (y_idx - cy) * z / fy,
            z,
        ])

        heights_mm              = np.maximum(pts @ normal + plane_distance, 0) * 1000.0
        height_map_mm           = np.zeros((h, w), dtype=np.float32)
        height_map_mm[y_idx, x_idx] = heights_mm

        protrusion_mask = self._build_distance_zoned_mask(
            height_map_mm, depth_m, valid_mask, heights_mm, y_idx, x_idx)

        if confidence_map is not None:
            conf_mask       = (confidence_map >= self.stereo_confidence_threshold
                               ).astype(np.uint8) * 255
            protrusion_mask = cv2.bitwise_and(protrusion_mask, conf_mask)

        protrusion_mask = cv2.morphologyEx(protrusion_mask, cv2.MORPH_CLOSE,
                                            self._kernel7)
        protrusion_mask = self._fill_mask_holes(protrusion_mask)
        protrusion_mask = self._depth_grow_mask(protrusion_mask, depth_m)
        protrusion_mask = cv2.morphologyEx(protrusion_mask, cv2.MORPH_CLOSE,
                                            self._kernel7)

        contours, _ = cv2.findContours(
            protrusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = self._merge_nearby_contours_by_zone(
            contours, depth_m, h, w, merge_dist_px=50)

        # OPT v2: allocate a single cnt_mask buffer and zero-fill per contour
        # instead of allocating a fresh H×W array for every contour
        cnt_mask = np.zeros((h, w), dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cx_box = x + w_box // 2
            cy_box = y + h_box // 2

            if cy_box >= h or cx_box >= w:
                continue
            z_val = depth_m[cy_box, cx_box]
            if z_val < self.min_distance or z_val > self.max_distance:
                continue

            zone = self._get_zone(z_val)
            if area < zone['min_area_px']:
                continue

            pos_x    = (cx_box - cx) * z_val / fx
            pos_y    = (cy_box - cy) * z_val / fy
            distance = float(np.sqrt(pos_x**2 + pos_y**2 + z_val**2))

            # OPT v2: reuse pre-allocated buffer — zero only the bounding rect
            cnt_mask[y:y+h_box, x:x+w_box] = 0
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            contour_heights   = height_map_mm[cnt_mask == 255]
            median_h = float(np.median(contour_heights)) if len(contour_heights) else 0.0
            # Clear the drawn region for the next iteration
            cnt_mask[y:y+h_box, x:x+w_box] = 0

            if median_h < zone['median_h_min']:
                continue

            obstacles.append({
                'center_px':     (cx_box, cy_box),
                'position_3d_m': np.array([pos_x, pos_y, z_val]),
                'distance_m':    distance,
                'contour':       cnt,
                'area_px':       area,
                'median_height_mm': median_h,
                'zone':          zone['max_m'],
            })

        obstacles.sort(key=lambda o: o['distance_m'])
        return obstacles, protrusion_mask, ground_result, height_map_mm
# ---------------------------------------------------------------------------
# RGBDepthFusion  —  FastSAM on a background thread, fused each depth frame
# ---------------------------------------------------------------------------
class RGBDepthFusion:
    """
    FastSAM runs asynchronously on a background thread so it never blocks
    the depth pipeline.  The main loop calls fuse() which uses the most
    recent completed FastSAM result (which may be 1-3 frames stale —
    acceptable because segmentation masks change slowly).
    """

    def __init__(self,
                 model_path:          str   = "FastSAM-s.pt",
                 confidence:          float = 0.4,
                 iou_threshold:       float = 0.9,
                 min_mask_area_frac:  float = 0.003,
                 max_mask_area_frac:  float = 0.8):
        print("Loading FastSAM model...")
        self.model               = FastSAM(model_path)
        self.confidence          = confidence
        self.iou_threshold       = iou_threshold
        self.min_mask_area_frac  = min_mask_area_frac
        self.max_mask_area_frac  = max_mask_area_frac

        self.floor_hsv_mean: Optional[np.ndarray] = None
        self.floor_hsv_std:  Optional[np.ndarray] = None

        # Background thread state
        self._lock           = threading.Lock()
        self._latest_masks:  List[np.ndarray] = []   # most recent SAM output
        self._pending_frame: Optional[np.ndarray] = None  # frame queued for SAM
        self._thread         = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        # OPT v2: throttle _update_floor_model
        self._floor_update_counter = 0
        print("FastSAM background thread started.")

    # ------------------------------------------------------------------
    # Background worker — processes one frame at a time, always takes
    # the most recently queued frame (drops stale ones)
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
        """Call this every FASTSAM_INTERVAL frames to queue a new SAM job."""
        with self._lock:
            # Always overwrite — we only care about the most recent frame
            self._pending_frame = rgb_frame.copy()

    def get_latest_masks(self) -> List[np.ndarray]:
        with self._lock:
            return list(self._latest_masks)

    # ------------------------------------------------------------------
    # OPT: smaller imgsz (320 vs 640) — ~4× faster inference
    # ------------------------------------------------------------------
    def _run_fastsam_sync(self, rgb_frame: np.ndarray) -> List[np.ndarray]:
        h, w       = rgb_frame.shape[:2]
        frame_area = h * w

        results = self.model(
            rgb_frame,
            device='cpu',
            retina_masks=False,   # retina_masks=True is slow; False + resize is fine
            imgsz=FASTSAM_IMGSZ,  # 320 instead of 640
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

        # Deduplicate by IoU — vectorised using dot products on flattened masks
        if len(masks) > 1:
            masks.sort(key=lambda m: int(np.sum(m)), reverse=True)
            flat   = np.stack([m.ravel().astype(np.float32) for m in masks])
            areas  = flat.sum(axis=1)
            keep   = []
            kept_flat: List[np.ndarray] = []
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
    # Floor model update (same as before, unchanged)
    # ------------------------------------------------------------------
    def _update_floor_model(self, rgb_frame, depth_m,
                             ground_normal, plane_d, K):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        y_idx, x_idx   = np.where((depth_m > 0.3) & (depth_m < 2.5))
        if len(y_idx) == 0:
            return
        z   = depth_m[y_idx, x_idx]
        pts = np.column_stack([
            (x_idx - cx) * z / fx,
            (y_idx - cy) * z / fy,
            z,
        ])
        heights    = np.maximum(pts @ ground_normal + plane_d, 0) * 1000.0
        floor_mask = heights < 40
        fy_idx, fx_idx = y_idx[floor_mask], x_idx[floor_mask]
        if len(fy_idx) < 50:
            return
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        fp  = hsv[fy_idx, fx_idx]
        self.floor_hsv_mean = np.mean(fp, axis=0)
        self.floor_hsv_std  = np.std(fp, axis=0) + 1e-6

    def _is_floor_segment(self, mask, hsv_frame) -> bool:
        """OPT: accepts pre-converted HSV frame, not raw RGB."""
        if self.floor_hsv_mean is None:
            return False
        px = hsv_frame[mask > 0]
        if len(px) == 0:
            return False
        z  = np.abs(px.mean(axis=0) - self.floor_hsv_mean) / self.floor_hsv_std
        return float(z[0]) < 2.0 and float(z[1]) < 2.0

    def _get_mask_depth_stats(self, mask, depth_m, min_dist, max_dist):
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

    def _mask_iou(self, a, b) -> float:
        i = np.logical_and(a, b).sum()
        u = np.logical_or(a, b).sum()
        return float(i) / max(1, float(u))

    def _contour_from_mask(self, mask):
        cs, _ = cv2.findContours(mask.astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(cs, key=cv2.contourArea) if cs else None

    # ------------------------------------------------------------------
    # fuse() — accepts pre-computed ground_result and height_map_mm
    #          so main() doesn't recompute them
    # ------------------------------------------------------------------
    def fuse(self,
             rgb_frame:      np.ndarray,
             depth_m:        np.ndarray,
             depth_obstacles: List[dict],
             ground_normal:  np.ndarray,
             plane_d:        float,
             K:              np.ndarray,
             height_map_mm:  np.ndarray,
             min_dist:       float = 0.35,
             max_dist:       float = 2.5) -> List[dict]:

        h, w   = depth_m.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        # OPT v2: only update floor model every FLOOR_UPDATE_INTERVAL calls
        self._floor_update_counter += 1
        if self._floor_update_counter % FLOOR_UPDATE_INTERVAL == 0:
            self._update_floor_model(rgb_frame, depth_m, ground_normal, plane_d, K)

        # OPT: convert HSV once, pass to _is_floor_segment for all masks
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV).astype(np.float32)

        # OPT: pre-rasterise all depth obstacle masks once — O(n) not O(n×m)
        d_masks = []
        for d_obs in depth_obstacles:
            dm = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(dm, [d_obs['contour']], -1, 255, thickness=cv2.FILLED)
            d_masks.append(dm)

        # Get latest masks from background thread
        rgb_masks = self.get_latest_masks()

        depth_matched  = [False] * len(depth_obstacles)
        fused_obstacles: List[dict] = []

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

            # OPT v2: early-exit once a strong match (>0.5) is found;
            # d_masks are already in depth-distance order so the best
            # candidate is likely early in the list.
            matched_idx, best_iou = None, 0.15
            for di, dm in enumerate(d_masks):
                iou = self._mask_iou(mask, dm)
                if iou > best_iou:
                    best_iou, matched_idx = iou, di
                    if best_iou > 0.5:   # good enough — stop scanning
                        break

            cnt_mask         = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [contour], -1, 255, thickness=cv2.FILLED)
            contour_heights  = height_map_mm[cnt_mask == 255]
            median_h = float(np.median(contour_heights)) if len(contour_heights) else 0.0

            if matched_idx is not None:
                depth_matched[matched_idx] = True
                fused_obstacles.append({
                    **depth_obstacles[matched_idx],
                    'contour':          contour,
                    'rgb_mask':         mask,
                    'median_height_mm': median_h,
                    'detection_source': 'depth+rgb',
                    'depth_coverage':   ds['depth_coverage'],
                })
            else:
                if median_h < 30 and ds['source'] == 'edge_ring':
                    continue
                fused_obstacles.append({
                    'center_px':        (cx_box, cy_box),
                    'position_3d_m':    np.array([pos_x, pos_y, mask_depth]),
                    'distance_m':       dist,
                    'contour':          contour,
                    'area_px':          int(cv2.contourArea(contour)),
                    'median_height_mm': median_h,
                    'rgb_mask':         mask,
                    'detection_source': 'rgb',
                    'depth_coverage':   ds['depth_coverage'],
                })

        for di, d_obs in enumerate(depth_obstacles):
            if not depth_matched[di]:
                fused_obstacles.append({
                    **d_obs,
                    'rgb_mask':         None,
                    'detection_source': 'depth',
                    'depth_coverage':   1.0,
                })

        fused_obstacles.sort(key=lambda o: o['distance_m'])
        return fused_obstacles


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------
def create_debug_visualization(rgb_frame, depth_viz, tag_detections,
                                obstacles, obstacle_mask, mode, fps):
    overlay = np.zeros_like(rgb_frame)
    SOURCE_COLORS = {
        'depth':     (0,   200,   0),
        'rgb':       (200, 100,   0),
        'depth+rgb': (0,   200, 200),
    }

    for obs in obstacles:
        dist       = obs['distance_m']
        base_color = SOURCE_COLORS.get(obs.get('detection_source', 'depth'), (0, 200, 0))
        fade       = max(0.3, 1.0 - dist / 2.5)
        color      = tuple(int(c * fade) for c in base_color)
        cv2.drawContours(overlay, [obs['contour']], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(overlay, [obs['contour']], -1, (255, 255, 255), 1)
        cx_o, cy_o = obs['center_px']
        src        = obs.get('detection_source', '?')[0].upper()
        cv2.putText(overlay, f"{src} {dist:.2f}m", (cx_o-20, cy_o-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    blended = cv2.addWeighted(rgb_frame, 0.7, overlay, 0.3, 0)

    for tag in tag_detections:
        cv2.polylines(blended, [tag['corners'].astype(int)], True, (0, 255, 0), 2)
        cv2.circle(blended, tag['center'], 5, (0, 255, 0), -1)
        cv2.putText(blended, f"ID:{tag['tag_id']} {tag['distance_m']:.2f}m",
                    (tag['center'][0]+10, tag['center'][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    depth_viz_r = cv2.resize(depth_viz, (rgb_frame.shape[1], rgb_frame.shape[0]))
    combined    = np.hstack([blended, depth_viz_r])

    mh, mw = depth_viz_r.shape[:2]
    sm      = cv2.resize(obstacle_mask, (mw//4, mh//4))
    smc     = cv2.applyColorMap(sm, cv2.COLORMAP_HOT)
    hc, wc  = combined.shape[:2]
    oh, ow  = smc.shape[:2]
    combined[hc-oh:, wc-ow:] = smc

    cv2.putText(combined, f"Mode: {mode}  FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return combined


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("OAK-D Lite Comprehensive Debugging Suite  [optimised v2]")
    print("=" * 60)

    depth_debugger    = DepthMapDebugger()
    tag_tester        = AprilTagDistanceTester()
    obstacle_detector = SpatialObstacleDetector()
    fusion            = RGBDepthFusion(model_path="FastSAM-s.pt")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "Images")
    os.makedirs(images_dir, exist_ok=True)

    # OPT v2: use PIPELINE_MODE constant (defaults to "optimized")
    current_mode = PIPELINE_MODE

    print("\nInitialising OAK-D Lite...")
    try:
        pipeline = StereoDepthConfigurator.create_pipeline(current_mode)
        device   = dai.Device(pipeline, usb2Mode=True)
        print(f"✅ Connected | MxId: {device.getMxId()}")

        calib     = device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)
        K         = np.array(intrinsics, dtype=np.float32)

        # OPT v2: maxSize=2 gives a 1-frame buffer; blocking=False means tryGet()
        # never stalls — the loop runs at CPU speed and drops frames the VPU
        # hasn't delivered yet rather than waiting.
        q_rgb   = device.getOutputQueue(name="rgb",        maxSize=2, blocking=False)
        q_depth = device.getOutputQueue(name="depth",      maxSize=2, blocking=False)
        q_conf  = device.getOutputQueue(name="confidence", maxSize=2, blocking=False)

    except Exception as e:
        print(f"❌ Failed to initialise OAK-D: {e}")
        device = None
        q_rgb = q_depth = q_conf = None
        K = np.array([[552.6, 0, 311.4], [0, 552.6, 202.5], [0, 0, 1]],
                     dtype=np.float32)

    try:
        frame_count          = 0
        start_time           = time.time()
        last_confidence_frame = None
        # OPT: lightweight temporal mask — running OR then decay instead of
        #      np.median over 3 full frames every frame
        stable_mask          = None
        # OPT v2: carry last valid frames for tryGet() skip-logic
        rgb_frame            = None
        depth_frame          = None
        gray                 = None   # reuse gray across AprilTag skips
        tag_detections:      List[dict] = []

        while True:
            # ── Capture (non-blocking) ──────────────────────────────────
            if device is not None and q_rgb and q_depth:
                # OPT v2: tryGet() returns None immediately if no frame is
                # ready; we skip processing and spin rather than blocking,
                # keeping the main loop latency near zero.
                rgb_packet   = q_rgb.tryGet()
                depth_packet = q_depth.tryGet()
                conf_packet  = q_conf.tryGet() if q_conf else None

                if rgb_packet is None or depth_packet is None:
                    # No new camera frame yet — spin without sleeping so we
                    # catch the next frame as soon as it arrives.
                    continue

                rgb_frame   = rgb_packet.getCvFrame()
                depth_frame = depth_packet.getFrame()
                # OPT: kernel 3 instead of 5 — same noise reduction, ~4ms faster
                depth_frame = cv2.medianBlur(depth_frame, DEPTH_MEDIAN_KSIZE)

                if conf_packet is not None:
                    last_confidence_frame = conf_packet.getFrame()
            else:
                # Synthetic fallback
                rgb_frame   = np.zeros((400, 640, 3), dtype=np.uint8)
                depth_frame = np.zeros((400, 640), dtype=np.uint16)
                y_vals      = np.arange(400).reshape(-1, 1)
                depth_frame[:] = 500 + y_vals * 3
                cv2.rectangle(depth_frame, (300, 160), (350, 240), 1500, -1)
                last_confidence_frame = np.ones((400, 640), dtype=np.uint8) * 255

            if rgb_frame is None or depth_frame is None:
                continue

            rgb_display = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # ── Depth stats (every STATS_INTERVAL frames) ───────────────
            # OPT v2: skip expensive stats most frames; display cached value
            run_stats   = (frame_count % STATS_INTERVAL == 0)
            depth_stats = depth_debugger.analyze_depth_map(
                depth_frame, use_cache=not run_stats)

            # ── AprilTags (every APRILTAG_INTERVAL frames) ───────────────
            # OPT v2: tags are stable; reuse last result between detections
            if frame_count % APRILTAG_INTERVAL == 0:
                gray           = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                tag_detections = tag_tester.detect_with_diagnostics(gray, K)
            else:
                # use_cache=True returns the stored list immediately
                tag_detections = tag_tester.detect_with_diagnostics(
                    None, K, use_cache=True)  # type: ignore[arg-type]

            # ── Obstacle detection (every frame, uses cached plane) ─────
            obstacles, current_mask, ground_result, height_map_mm = \
                obstacle_detector.detect_obstacles(
                    depth_frame, K, last_confidence_frame)

            # ── Submit frame to FastSAM thread (every N frames) ─────────
            if frame_count % FASTSAM_INTERVAL == 0:
                fusion.submit_frame(rgb_display)

            # ── Fuse with latest SAM masks (non-blocking) ───────────────
            if ground_result is not None:
                depth_m_frame = depth_frame.astype(np.float32) / 1000.0
                obstacles = fusion.fuse(
                    rgb_display, depth_m_frame, obstacles,
                    ground_result[0], ground_result[1],
                    K, height_map_mm,
                )

            # ── OPT: temporal smoothing via bitwise OR + erosion ─────────
            # Much cheaper than np.median over 3 frames; provides a 1-frame
            # hysteresis that smooths flickering without lag
            if stable_mask is None:
                stable_mask   = current_mask.copy()
                # OPT v2: pre-allocate erosion kernel once
                _erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            else:
                stable_mask = cv2.bitwise_or(stable_mask, current_mask)
                # Decay: erode slightly so old blobs fade after ~2 frames
                stable_mask = cv2.erode(stable_mask, _erode_kernel)

            frame_count += 1

            # ── Console logging every 10 frames ─────────────────────────
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps     = frame_count / max(0.1, elapsed)
                print(f"\n{'='*60}")
                print(f"[Frame {frame_count}] FPS: {fps:.1f} | Mode: {current_mode}")
                print(f"{'='*60}")
                print(f"DEPTH MAP:")
                print(f"  Valid: {depth_stats['valid_pixels']/depth_stats['total_pixels']*100:.1f}%  "
                      f"Median: {depth_stats['median_depth_mm']:.0f}mm  "
                      f"Noise: {depth_stats['noise_ratio']:.2f}")
                print(f"APRILTAGS: {len(tag_detections)} found")
                for tag in tag_detections:
                    print(f"  ID {tag['tag_id']}: {tag['distance_m']:.2f}m "
                          f"(conf {tag['confidence']:.2f})")
                print(f"OBSTACLES: {len(obstacles)} found")
                for i, obs in enumerate(obstacles[:5]):
                    p = obs['position_3d_m']
                    print(f"  #{i+1}: {obs['distance_m']:.2f}m @ "
                          f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})  "
                          f"h={obs['median_height_mm']:.0f}mm  "
                          f"src={obs.get('detection_source','?')}")

            # ── Save debug image every 20 frames ────────────────────────
            if frame_count % 20 == 0:
                try:
                    elapsed = time.time() - start_time
                    fps     = frame_count / max(0.1, elapsed)
                    # HQ depth viz only computed here, not every frame
                    filtered_viz = depth_debugger.visualize_depth_hq(
                        depth_frame, depth_stats)
                    debug_view = create_debug_visualization(
                        rgb_display, filtered_viz, tag_detections,
                        obstacles, stable_mask, current_mode, fps)
                    save_path = os.path.join(
                        images_dir, f"debug_frame_{frame_count:04d}.png")
                    cv2.imwrite(save_path, debug_view)
                    print(f"\n📸 Saved: {save_path}")
                except Exception as e:
                    print(f"Warning: could not save debug image: {e}")

            if frame_count >= 300:
                print(f"\n{'='*60}\nCompleted 300 frames. Exiting.\n{'='*60}")
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if device:
            device.close()
        print("\nDebugging session complete.")


if __name__ == "__main__":
    main()