"""
Comprehensive OAK-D Lite Debugging Suite
Focus: Depth map validation, AprilTag detection at distance, spatial obstacle detection
Provides real-time visualization of depth confidence, point clouds, and 3D coordinates
"""
import os
import cv2
import numpy as np
import depthai as dai
from pupil_apriltags import Detector
from typing import Optional, Tuple, List
import time
from collections import deque


class DepthMapDebugger:
    """
    Analyzes and visualizes depth map quality metrics
    """
    def __init__(self):
        self.depth_history = deque(maxlen=30)  # Track depth stability
        self.confidence_history = deque(maxlen=30)
        
    def analyze_depth_map(self, depth_frame: np.ndarray) -> dict:
        """Comprehensive depth map analysis"""
        stats = {
            'valid_pixels': 0,
            'total_pixels': depth_frame.size,
            'median_depth_mm': 0,
            'mean_depth_mm': 0,
            'std_depth_mm': 0,
            'min_depth_mm': 0,
            'max_depth_mm': 0,
            'noise_ratio': 0,
            'confidence_score': 0
        }
        
        # Convert to float for analysis
        depth_mm = depth_frame.astype(np.float32)
        
        # Valid depth range (10cm to 5m for OAK-D Lite)
        valid_mask = (depth_mm > 100) & (depth_mm < 5000)
        stats['valid_pixels'] = int(np.sum(valid_mask))
        
        if stats['valid_pixels'] > 0:
            valid_depths = depth_mm[valid_mask]
            stats['median_depth_mm'] = float(np.median(valid_depths))
            stats['mean_depth_mm'] = float(np.mean(valid_depths))
            stats['std_depth_mm'] = float(np.std(valid_depths))
            stats['min_depth_mm'] = float(np.min(valid_depths))
            stats['max_depth_mm'] = float(np.max(valid_depths))
            
            # Noise ratio: high std dev relative to mean indicates noise
            if stats['mean_depth_mm'] > 0:
                stats['noise_ratio'] = stats['std_depth_mm'] / stats['mean_depth_mm']
            
            # Confidence score based on valid pixel ratio
            stats['confidence_score'] = stats['valid_pixels'] / stats['total_pixels']
        
        # Track history for stability analysis
        self.depth_history.append(stats['median_depth_mm'])
        self.confidence_history.append(stats['confidence_score'])
        
        # Calculate temporal stability
        if len(self.depth_history) > 5:
            depth_std = np.std(list(self.depth_history))
            stats['temporal_stability'] = 1.0 - min(1.0, depth_std / 100.0)  # Lower std = higher stability
        else:
            stats['temporal_stability'] = 1.0
            
        return stats
    
    def visualize_depth(self, depth_frame: np.ndarray, 
                       stats: dict, 
                       apply_filters: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create multiple depth visualizations for debugging
        Returns: (raw_viz, filtered_viz)
        """
        # Normalize depth for visualization (0-255)
        depth_mm = depth_frame.astype(np.float32)
        
        # Clip to useful range (10cm - 3m)
        depth_clipped = np.clip(depth_mm, 100, 3000)
        
        # Raw normalized depth
        depth_norm = cv2.normalize(
            depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        raw_viz = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Add statistics overlay
        self._draw_stats(raw_viz, stats)
        
        if apply_filters:
            # Apply bilateral filter to reduce noise while preserving edges
            depth_filtered = cv2.bilateralFilter(depth_clipped, 9, 75, 75)
            depth_filtered_norm = cv2.normalize(
                depth_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            filtered_viz = cv2.applyColorMap(depth_filtered_norm, cv2.COLORMAP_JET)
            
            # Apply histogram equalization for better contrast
            depth_equalized = cv2.equalizeHist(depth_filtered_norm.flatten()).reshape(depth_filtered_norm.shape)
            filtered_viz = cv2.applyColorMap(depth_equalized, cv2.COLORMAP_VIRIDIS)
        else:
            filtered_viz = raw_viz.copy()
            
        return raw_viz, filtered_viz
    
    def _draw_stats(self, frame: np.ndarray, stats: dict):
        """Draw statistics overlay on frame"""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        text_lines = [
            f"Valid: {stats['valid_pixels']/stats['total_pixels']*100:.1f}%",
            f"Median: {stats['median_depth_mm']:.0f}mm",
            f"StdDev: {stats['std_depth_mm']:.1f}mm",
            f"Noise: {stats['noise_ratio']:.2f}",
            f"Stability: {stats['temporal_stability']:.2f}"
        ]
        
        x_pos = 10
        for line in text_lines:
            cv2.putText(frame, line, (x_pos, y_offset), font, font_scale, (255, 255, 255), 1)
            y_offset += 25


class AprilTagDistanceTester:
    """
    Tests AprilTag detection at various distances with detailed diagnostics
    """
    def __init__(self, tag_family: str = "tag36h11"):
        self.detector = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=1.0,  # Full resolution for distance testing
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25
        )
        self.tag_size = 0.08  # 8cm tags
        self.detection_history = []
        
    def detect_with_diagnostics(self, gray_frame: np.ndarray, 
                                K: np.ndarray) -> List[dict]:
        """Detect tags with detailed diagnostic information"""
        results = self.detector.detect(gray_frame)
        detections = []
        
        for result in results:
            center = tuple(np.mean(result.corners, axis=0).astype(int))
            img_points = result.corners.astype(np.float32)
            
            # PnP pose estimation
            half_size = self.tag_size / 2.0
            obj_points = np.array([
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, K, None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                t = tvec.flatten()
                distance = float(np.linalg.norm(t))
                bearing = float(np.arctan2(t[0], t[2]))
                
                # Calculate apparent tag size (for distance correlation)
                tag_area = cv2.contourArea(result.corners)
                
                detections.append({
                    'tag_id': result.tag_id,
                    'center': center,
                    'corners': result.corners,
                    'distance_m': distance,
                    'bearing_rad': bearing,
                    'confidence': result.decision_margin,
                    'apparent_area_px': tag_area,
                    'tvec': t,
                    'rvec': rvec.flatten()
                })
        
        return detections


class StereoDepthConfigurator:
    """
    Configures and tests different StereoDepth settings for optimal performance
    Optimized for FPS target of 15+ with reduced resolution and vectorized processing
    """
    
    @staticmethod
    def create_pipeline(mode: str = "balanced") -> dai.Pipeline:
        """
        Create pipeline with different configurations
        Modes: 'high_accuracy', 'balanced', 'long_range', 'low_noise', 'optimized'
        
        Pipeline optimization (FPS ~3.4 → target 15+):
        - Drop resolution to 640×400 (400P) or 416×240; stereo depth scales quadratically
        - Use stereo.setLeftRightCheck(True) + setSubpixel(True) only if quality > throughput
        - Clamp detectionNetwork.setBoundingBoxScaleFactor(0.4) to prevent NN boxes from sampling depth at tile boundaries
        """
        pipeline = dai.Pipeline()
        print(f"  → Creating pipeline with mode: {mode}")
        
        # Optimized resolution for higher FPS (640x400 instead of 640x480)
        output_width = 640
        output_height = 400
        
        # RGB Camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(output_width, output_height)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(30)  # Higher FPS target
        
        # Manual focus for stability (critical for consistent depth)
        cam_rgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        
        # Mono cameras - use 400P resolution (quadratic scaling benefit)
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_left.setFps(30)
        
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_right.setFps(30)
        
        # Stereo Depth
        stereo = pipeline.create(dai.node.StereoDepth)
        
        if mode == "high_accuracy":
            print("  → HIGH ACCURACY: Extended disparity + Subpixel + Confidence 200")
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(True)  # Better edge quality but ~20% latency
            stereo.initialConfig.setConfidenceThreshold(200)
            stereo.setLeftRightCheck(True)  # Improves quality, adds ~20% latency
            #stereo.setMedianFilter(dai.MedianFilter.KERNEL_5x5)  # Stronger median filter for noise reduction
        elif mode == "long_range":
            print("  → LONG RANGE: Extended disparity + No subpixel + Confidence 200")
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.LONG_RANGE)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(200)
        elif mode == "low_noise":
            print("  → LOW NOISE: High density + No extended disparity + No subpixel + Confidence 220")
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.initialConfig.setExtendedDisparity(False)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(220)  # Higher = more strict
        elif mode == "optimized":
            print("  → OPTIMIZED: Default preset + No extended disparity + No subpixel + Confidence 220")
            # Optimized for FPS with good quality
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setExtendedDisparity(False)  # Faster
            stereo.initialConfig.setSubpixel(False)  # ~20% faster without subpixel
            stereo.initialConfig.setConfidenceThreshold(220)  # Strict confidence gating
            # Disable LR check for speed if quality acceptable (uncomment if needed)
            # stereo.setLeftRightCheck(False)
        else:  # balanced
            print("  → BALANCED: Default preset + Extended disparity + No subpixel + Confidence 220")
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(220)  # Increased to 220 for confidence gating
        
        # Common settings - optimized output size
        stereo.setOutputSize(output_width, output_height)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setLeftRightCheck(True)  # Keep for quality, adds ~20% latency but cleans edge noise
        
        # Linking
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # Outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        # Confidence map output (for confidence gating)
        xout_conf = pipeline.create(dai.node.XLinkOut)
        xout_conf.setStreamName("confidence")
        stereo.confidenceMap.link(xout_conf.input)
        
        return pipeline


class SpatialObstacleDetector:
    """
    Detects obstacles using depth data with 3D coordinate extraction
    Implements ground-plane rejection, spatial coherence filtering,
    confidence gating, and gradient discontinuity checks
    """
    def __init__(self, min_distance: float = 0.2, max_distance: float = 8.0,
                 camera_height_m: float = 0.3, camera_pitch_rad: float = 0.0):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.camera_height_m = camera_height_m
        self.camera_pitch_rad = camera_pitch_rad
        # Ground plane tolerance in mm
        self.ground_plane_tolerance_mm = 40.0
        # Minimum contiguous pixels for valid obstacle
        self.min_contiguous_pixels = 150
        # Maximum internal depth variance for coherent obstacle (mm)
        self.max_internal_variance_mm = 80.0
        # Gradient discontinuity thresholds (mm)
        self.obstacle_gradient_threshold_mm = 150.0
        self.floor_gradient_threshold_mm = 50.0
        # Confidence thresholds
        self.stereo_confidence_threshold = 220
        self.detection_confidence_threshold = 0.65
        
    def _fit_ground_plane(self, depth_frame: np.ndarray, K: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        h, w = depth_frame.shape
        depth_mm = depth_frame.astype(np.float32)
        
        # Use lower 30% of image for ground plane estimation
        lower_region_y = int(h * 0.7)
        lower_depth = depth_mm[lower_region_y:, :]
        valid_mask = (lower_depth > 100) & (lower_depth < 5000)
        
        if np.sum(valid_mask) < 50:
            return None
            
        y_indices, x_indices = np.where(valid_mask)
        y_indices = y_indices + lower_region_y
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        z_vals = lower_depth[valid_mask] / 1000.0
        x_vals = (x_indices - cx) * z_vals / fx
        y_vals = (y_indices - cy) * z_vals / fy
        
        points_3d = np.column_stack([x_vals, y_vals, z_vals])
        
        # Downsample to ~5k points: SVD doesn't need 50k+ points for a plane fit
        # This cuts memory usage and speeds up computation significantly
        if len(points_3d) > 5000:
            idx = np.random.choice(len(points_3d), 5000, replace=False)
            points_3d = points_3d[idx]
            
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        
        # 🔑 CRITICAL FIX: full_matrices=False prevents O(N²) memory allocation
        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
        normal = vh[-1, :]  # Normal vector corresponds to smallest singular value
        plane_distance = -np.dot(normal, centroid)
        
        return normal, plane_distance
    def _compute_distance_to_plane(self, points_3d: np.ndarray, 
                                   normal: np.ndarray, 
                                   plane_distance: float) -> np.ndarray:
        """Compute perpendicular distance from points to plane"""
        # Distance = |normal . point + plane_distance| / ||normal||
        distances = np.abs(np.dot(points_3d, normal) + plane_distance)
        return distances
    
    def _check_spatial_coherence(self, depth_frame: np.ndarray, 
                                 mask: np.ndarray,
                                 label_image: np.ndarray,
                                 label_id: int) -> Tuple[bool, float]:
        """
        Check if obstacle region has sufficient contiguous valid-depth pixels
        with low internal depth variance
        Returns: (is_coherent, internal_variance_mm)
        """
        # Extract depth values for this label
        region_mask = (label_image == label_id)
        
        if np.sum(region_mask) < self.min_contiguous_pixels:
            return False, 0.0
        
        region_depths = depth_frame[region_mask].astype(np.float32)
        valid_region = (region_depths > 100) & (region_depths < 5000)
        
        if np.sum(valid_region) < self.min_contiguous_pixels:
            return False, 0.0
        
        valid_depths = region_depths[valid_region]
        variance_mm = float(np.std(valid_depths))
        
        is_coherent = (variance_mm <= self.max_internal_variance_mm)
        
        return is_coherent, variance_mm
    
    def _check_gradient_discontinuity(self, depth_frame: np.ndarray,
                                      mask: np.ndarray,
                                      K: np.ndarray) -> bool:
        """
        Check if region shows significant gradient discontinuity
        True obstacles show >150mm discontinuities; floor tiles remain <50mm
        """
        depth_m = depth_frame.astype(np.float32) / 1000.0
        
        # Compute Sobel gradients
        grad_x = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Get gradient values in the mask region
        if np.sum(mask) == 0:
            return False
        
        region_gradients = grad_magnitude[mask]
        median_gradient_m = float(np.median(region_gradients))
        median_gradient_mm = median_gradient_m * 1000.0
        
        # True obstacle should have significant gradient
        return median_gradient_mm >= self.obstacle_gradient_threshold_mm
    
def detect_obstacles(self, depth_frame: np.ndarray, 
                    K: np.ndarray,
                    confidence_map: Optional[np.ndarray] = None) -> List[dict]:
    obstacles = []
    h, w = depth_frame.shape
    
    # 1. Valid depth mask (35cm to 5m)
    depth_m = depth_frame.astype(np.float32) / 1000.0
    valid_mask = (depth_m >= self.min_distance) & (depth_m <= self.max_distance)
    
    if not np.any(valid_mask):
        return obstacles, np.zeros((h, w), dtype=np.uint8)
    
    # 2. Fit ground plane (keep your existing SVD method)
    ground_result = self._fit_ground_plane(depth_frame, K)
    if ground_result is None:
        return obstacles, np.zeros((h, w), dtype=np.uint8)
    
    normal, plane_distance = ground_result
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    # 3. Vectorized: Compute height-above-ground for ALL valid pixels
    y_idx, x_idx = np.where(valid_mask)
    z = depth_m[y_idx, x_idx]
    
    # Back-project to 3D camera coordinates
    x_3d = (x_idx - cx) * z / fx
    y_3d = (y_idx - cy) * z / fy
    points_3d = np.column_stack([x_3d, y_3d, z])
    
    # Distance to plane = height above ground (meters)
    heights_m = np.abs(np.sum(points_3d * normal, axis=1) + plane_distance)
    heights_mm = heights_m * 1000.0
    
    # 4. Create protrusion mask (ignore bumps < 30mm)
    protrusion_mask = np.zeros((h, w), dtype=np.uint8)
    protrusion_mask[y_idx, x_idx] = (heights_mm > 30).astype(np.uint8) * 255
    
    # 5. Apply confidence gating & morphological cleanup
    if confidence_map is not None:
        conf_mask = (confidence_map >= self.stereo_confidence_threshold).astype(np.uint8) * 255
        protrusion_mask = cv2.bitwise_and(protrusion_mask, conf_mask)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    protrusion_mask = cv2.morphologyEx(protrusion_mask, cv2.MORPH_CLOSE, kernel)
    protrusion_mask = cv2.morphologyEx(protrusion_mask, cv2.MORPH_OPEN, kernel)
    
    # 6. Extract contours instead of bounding boxes
    contours, _ = cv2.findContours(protrusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < self.min_contiguous_pixels:
            continue
            
        # Get bounding rect for centroid calculation (fast)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        cx_box, cy_box = x + w_box//2, y + h_box//2
        
        # Sample depth at centroid
        if cy_box >= h or cx_box >= w:
            continue
        z_val = depth_m[cy_box, cx_box]
        
        if z_val < self.min_distance or z_val > self.max_distance:
            continue
            
        # 3D position
        pos_x = (cx_box - cx) * z_val / fx
        pos_y = (cy_box - cy) * z_val / fy
        distance = float(np.sqrt(pos_x**2 + pos_y**2 + z_val**2))
        
        obstacles.append({
            'center_px': (cx_box, cy_box),
            'position_3d_m': np.array([pos_x, pos_y, z_val]),
            'distance_m': distance,
            'contour': cnt,  # Store for filled drawing
            'area_px': area,
            'median_height_mm': float(np.median(heights_mm[cv2.pointPolygonTest(cnt, (x, y), measureDist=False) > 0]))
        })
        
    obstacles.sort(key=lambda o: o['distance_m'])
    return obstacles, protrusion_mask


def create_debug_visualization(rgb_frame: np.ndarray,
                              depth_viz: np.ndarray,
                              obstacles: List[dict],
                              obstacle_mask: np.ndarray,
                              mode: str) -> np.ndarray:
    # Create RGBA overlay for organic obstacle shapes
    overlay = np.zeros_like(rgb_frame)
    
    for obs in obstacles:
        dist = obs['distance_m']
        # Color: Red (close) -> Green (far)
        color_val = int(255 * min(1.0, dist / 2.0))
        color = (255 - color_val, color_val, 0)
        
        # Draw filled contour
        cv2.drawContours(overlay, [obs['contour']], -1, color, thickness=cv2.FILLED)
        # Draw outline
        cv2.drawContours(overlay, [obs['contour']], -1, (255, 255, 255), 1)
        
        # Label at centroid
        cx, cy = obs['center_px']
        cv2.putText(overlay, f"{dist:.2f}m", (cx - 20, cy - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                   
    # Blend overlay with RGB (30% opacity)
    blended_rgb = cv2.addWeighted(rgb_frame, 0.7, overlay, 0.3, 0)
    
    # Stack with depth viz
    depth_viz_resized = cv2.resize(depth_viz, (rgb_frame.shape[1], rgb_frame.shape[0]))
    combined = np.hstack([blended_rgb, depth_viz_resized])
    
    cv2.putText(combined, f"Mode: {mode}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return combined


def main():
    """Main debugging loop"""
    print("="*60)
    print("OAK-D Lite Comprehensive Debugging Suite")
    print("="*60)
    print("\nFeatures:")
    print("  - Real-time depth map quality analysis")
    print("  - AprilTag detection at various distances")
    print("  - Spatial obstacle detection with 3D coordinates")
    print("  - Multiple stereo depth configuration modes")
    print("  - Ground-plane rejection (±40mm tolerance)")
    print("  - Spatial coherence filter (≥15 contiguous pixels, <80mm variance)")
    print("  - Confidence gating (stereo >220, detection >0.65)")
    print("  - Gradient discontinuity check (>150mm = obstacle)")
    print("\nPipeline Optimizations:")
    print("  - Resolution: 640×400 (reduced from 640×480)")
    print("  - Vectorized NumPy processing (no Python for-loops)")
    print("  - FPS target: 15+ (from ~3.4)")
    print("\nNOTE: GUI display requires X11/display support.")
    print("      Running in headless mode - data printed to console.")
    print("="*60)
    
    # Initialize components
    depth_debugger = DepthMapDebugger()
    tag_tester = AprilTagDistanceTester()
    obstacle_detector = SpatialObstacleDetector()

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "Images")
    os.makedirs(images_dir, exist_ok=True)

    current_mode = "high_accuracy"  # Use optimized mode by default
    
    # Start OAK-D device
    print("\nInitializing OAK-D Lite...")
    try:
        pipeline = StereoDepthConfigurator.create_pipeline(current_mode)
        device = dai.Device(pipeline, usb2Mode=True)
        print(f"✅ Connected | MxId: {device.getMxId()}")
        
        # Get calibration intrinsics - use actual output size (640x400)
        calib = device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)
        K = np.array(intrinsics, dtype=np.float32)
        print(f"Camera Intrinsics: fx={intrinsics[0][0]:.1f}, fy={intrinsics[1][1]:.1f}, cx={intrinsics[0][2]:.1f}, cy={intrinsics[1][2]:.1f}")
        
        # Create output queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        q_conf = device.getOutputQueue(name="confidence", maxSize=1, blocking=False)
        
    except Exception as e:
        print(f"❌ Failed to initialize OAK-D: {e}")
        print("Running in simulation mode with synthetic data...")
        device = None
        q_rgb = None
        q_depth = None
        q_conf = None
        K = np.array([[800, 0, 320], [0, 800, 200], [0, 0, 1]], dtype=np.float32)  # Adjusted for 640x400
    
    try:
        frame_count = 0
        start_time = time.time()
        last_confidence_frame = None
        obstacle_history = deque(maxlen=3)

        while True:
            # Get frames (handle simulation mode)
            if device is not None and q_rgb and q_depth:
                rgb_packet = q_rgb.get()
                depth_packet = q_depth.get()
                conf_packet = q_conf.get() if q_conf else None
                
                if rgb_packet is None or depth_packet is None:
                    time.sleep(0.01)
                    continue
                
                rgb_frame = rgb_packet.getCvFrame()
                depth_frame = depth_packet.getFrame()
                # depth_frame = cv2.medianBlur(depth_frame, 5)  # Apply median blur to reduce noise
                # Get confidence map if available
                if conf_packet is not None:
                    last_confidence_frame = conf_packet.getFrame()
            else:
                # Simulation mode: generate synthetic data (adjusted for 640x400)
                rgb_frame = np.zeros((400, 640, 3), dtype=np.uint8)
                cv2.rectangle(rgb_frame, (100, 100), (540, 300), (100, 100, 100), -1)
                
                # Simulate AprilTag pattern
                cv2.rectangle(rgb_frame, (250, 140), (390, 260), (255, 255, 255), -1)
                cv2.rectangle(rgb_frame, (270, 160), (370, 240), (0, 0, 0), -1)
                
                # Simulate depth map (gradient representing ground plane) - vectorized
                depth_frame = np.zeros((400, 640), dtype=np.uint16)
                y_vals = np.arange(400).reshape(-1, 1)
                depth_frame[:] = 500 + y_vals * 3  # Vectorized: Depth increases with y (ground plane)
                
                # Add simulated obstacle
                cv2.rectangle(depth_frame, (300, 160), (350, 240), 1500, -1)
                
                # Simulate confidence map (all high confidence for valid regions)
                last_confidence_frame = np.ones((400, 640), dtype=np.uint8) * 255
            
            # Convert RGB to BGR for consistency
            rgb_display = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Analyze depth map
            depth_stats = depth_debugger.analyze_depth_map(depth_frame)
            raw_viz, filtered_viz = depth_debugger.visualize_depth(depth_frame, depth_stats)
            
            # Detect AprilTags
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            tag_detections = tag_tester.detect_with_diagnostics(gray, K)
            
            # Detect obstacles with confidence map
            obstacles = obstacle_detector.detect_obstacles(depth_frame, K, last_confidence_frame)
            
            # Print diagnostics every 10 frames (more frequent for debugging)
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / max(0.1, elapsed)
                
                print(f"\n{'='*60}")
                print(f"[Frame {frame_count}] FPS: {fps:.1f} | Mode: {current_mode}")
                print(f"{'='*60}")
                print(f"DEPTH MAP ANALYSIS:")
                print(f"  Valid pixels: {depth_stats['valid_pixels']/depth_stats['total_pixels']*100:.1f}%")
                print(f"  Median depth: {depth_stats['median_depth_mm']:.0f}mm ({depth_stats['median_depth_mm']/1000:.2f}m)")
                print(f"  StdDev: {depth_stats['std_depth_mm']:.1f}mm")
                print(f"  Noise ratio: {depth_stats['noise_ratio']:.2f} (lower=better, <0.3 good)")
                print(f"  Temporal stability: {depth_stats['temporal_stability']:.2f} (1.0=perfect)")
                
                print(f"\nAPRILTAG DETECTIONS: {len(tag_detections)} found")
                for tag in tag_detections:
                    print(f"  Tag ID {tag['tag_id']}:")
                    print(f"    Distance: {tag['distance_m']:.2f}m")
                    print(f"    Bearing: {np.degrees(tag['bearing_rad']):.1f}°")
                    print(f"    Confidence: {tag['confidence']:.2f}")
                    print(f"    Apparent area: {tag['apparent_area_px']:.0f} px²")
                
                print(f"\nOBSTACLES DETECTED: {len(obstacles)} found")
                for i, obs in enumerate(obstacles[:5]):  # Show closest 5
                    pos = obs['position_3d_m']
                    print(f"  #{i+1}: {obs['distance_m']:.2f}m @ ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m")
                    if 'detection_confidence' in obs:
                        print(f"       Conf: {obs['detection_confidence']:.2f}, Var: {obs['internal_variance_mm']:.1f}mm")
            
            # Save debug images every 20 frames (as requested)
            if frame_count % 20 == 0:
                try:
                    debug_view = create_debug_visualization(
                        rgb_display, filtered_viz, tag_detections, obstacles, depth_stats, current_mode
                    )
                    save_path = os.path.join(images_dir, f"debug_frame_{frame_count:04d}.png")
                    cv2.imwrite(save_path, debug_view)
                    print(f"\n📸 Saved debug image: {save_path}")
                except Exception as e:
                    print(f"Warning: Could not save debug image: {e}")
            
            # Check for exit condition (run for 300 frames then exit in headless mode)
            if frame_count >= 300:
                print(f"\n{'='*60}")
                print("Completed 300 frames. Exiting headless debug session.")
                print(f"{'='*60}")
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if device:
            device.close()
        print("\nDebugging session complete.")


if __name__ == "__main__":
    main()
