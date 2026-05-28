"""
Comprehensive OAK-D Lite Debugging Suite
Focus: Depth map validation, AprilTag detection at distance, spatial obstacle detection
Provides real-time visualization of depth confidence, point clouds, and 3D coordinates
"""

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
    """
    
    @staticmethod
    def create_pipeline(mode: str = "balanced") -> dai.Pipeline:
        """
        Create pipeline with different configurations
        Modes: 'high_accuracy', 'balanced', 'long_range', 'low_noise'
        """
        pipeline = dai.Pipeline()
        
        # RGB Camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(15)
        
        # Manual focus for stability (critical for consistent depth)
        cam_rgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        
        # Mono cameras
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        
        # Stereo Depth
        stereo = pipeline.create(dai.node.StereoDepth)
        
        if mode == "high_accuracy":
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(False)
        elif mode == "long_range":
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.LONG_RANGE)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(False)
        elif mode == "low_noise":
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.initialConfig.setExtendedDisparity(False)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(220)  # Higher = more strict
        else:  # balanced
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setExtendedDisparity(True)
            stereo.initialConfig.setSubpixel(False)
            stereo.initialConfig.setConfidenceThreshold(200)
        
        # Common settings
        stereo.setOutputSize(640, 480)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setLeftRightCheck(True)
        
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
        
        # Optional: Confidence map output
        xout_conf = pipeline.create(dai.node.XLinkOut)
        xout_conf.setStreamName("confidence")
        stereo.confidenceMap.link(xout_conf.input)
        
        return pipeline


class SpatialObstacleDetector:
    """
    Detects obstacles using depth data with 3D coordinate extraction
    """
    def __init__(self, min_distance: float = 0.1, max_distance: float = 5.0):
        self.min_distance = min_distance
        self.max_distance = max_distance
        
    def detect_obstacles(self, depth_frame: np.ndarray, 
                        K: np.ndarray) -> List[dict]:
        """
        Detect obstacles as depth discontinuities
        Returns list of obstacles with 3D positions
        """
        obstacles = []
        h, w = depth_frame.shape
        
        # Convert to meters
        depth_m = depth_frame.astype(np.float32) / 1000.0
        
        # Valid depth mask
        valid_mask = (depth_m > self.min_distance) & (depth_m < self.max_distance)
        
        if not np.any(valid_mask):
            return obstacles
        
        # Simple edge-based obstacle detection
        # Compute depth gradients
        grad_x = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold gradients to find depth discontinuities
        grad_threshold = 0.5  # Adjust based on scene
        obstacle_edges = grad_magnitude > grad_threshold
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        obstacle_edges = cv2.morphologyEx(
            obstacle_edges.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel
        )
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            obstacle_edges, connectivity=8
        )
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:  # Minimum obstacle size
                continue
            
            centroid_u, centroid_v = centroids[i]
            z = depth_m[int(centroid_v), int(centroid_u)]
            
            if z <= self.min_distance or z >= self.max_distance:
                continue
            
            # Convert to 3D camera coordinates
            x = (centroid_u - cx) * z / fx
            y = (centroid_v - cy) * z / fy
            
            obstacles.append({
                'center_px': (int(centroid_u), int(centroid_v)),
                'position_3d_m': np.array([x, y, z]),
                'distance_m': float(np.linalg.norm([x, y, z])),
                'size_px': (stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                'area_px': area
            })
        
        # Sort by distance
        obstacles.sort(key=lambda o: o['distance_m'])
        
        return obstacles


def create_debug_visualization(rgb_frame: np.ndarray,
                              depth_viz: np.ndarray,
                              tag_detections: List[dict],
                              obstacles: List[dict],
                              depth_stats: dict,
                              mode: str) -> np.ndarray:
    """Create comprehensive debug visualization"""
    h, w = rgb_frame.shape[:2]
    
    # Resize depth viz to match RGB
    depth_viz_resized = cv2.resize(depth_viz, (w, h))
    
    # Side-by-side display
    combined = np.hstack([rgb_frame, depth_viz_resized])
    
    # Draw tag detections
    for tag in tag_detections:
        # Draw corners
        corners = tag['corners'].astype(int)
        cv2.polylines(combined, [corners], True, (0, 255, 0), 2)
        
        # Draw center
        center = tag['center']
        cv2.circle(combined, center, 5, (0, 255, 0), -1)
        
        # Draw distance and ID
        label = f"ID:{tag['tag_id']} {tag['distance_m']:.2f}m"
        cv2.putText(combined, label, (center[0]+10, center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw obstacles
    for obs in obstacles:
        pt = obs['center_px']
        dist = obs['distance_m']
        
        # Color based on distance (red=close, green=far)
        color_val = int(255 * min(1.0, dist / 2.0))
        color = (255 - color_val, color_val, 0)
        
        cv2.circle(combined, pt, 8, color, -1)
        cv2.putText(combined, f"{dist:.2f}m", (pt[0]+10, pt[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add mode indicator
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
    print("\nNOTE: GUI display requires X11/display support.")
    print("      Running in headless mode - data printed to console.")
    print("="*60)
    
    # Initialize components
    depth_debugger = DepthMapDebugger()
    tag_tester = AprilTagDistanceTester()
    obstacle_detector = SpatialObstacleDetector()
    
    current_mode = "balanced"
    
    # Start OAK-D device
    print("\nInitializing OAK-D Lite...")
    try:
        pipeline = StereoDepthConfigurator.create_pipeline(current_mode)
        device = dai.Device(pipeline, usb2Mode=False)
        print(f"✅ Connected | MxId: {device.getMxId()}")
        
        # Get calibration intrinsics
        calib = device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
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
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Get frames (handle simulation mode)
            if device is not None and q_rgb and q_depth:
                rgb_packet = q_rgb.get()
                depth_packet = q_depth.get()
                
                if rgb_packet is None or depth_packet is None:
                    time.sleep(0.01)
                    continue
                
                rgb_frame = rgb_packet.getCvFrame()
                depth_frame = depth_packet.getFrame()
            else:
                # Simulation mode: generate synthetic data
                rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(rgb_frame, (100, 100), (540, 380), (100, 100, 100), -1)
                
                # Simulate AprilTag pattern
                cv2.rectangle(rgb_frame, (250, 180), (390, 320), (255, 255, 255), -1)
                cv2.rectangle(rgb_frame, (270, 200), (370, 300), (0, 0, 0), -1)
                
                # Simulate depth map (gradient representing ground plane)
                depth_frame = np.zeros((480, 640), dtype=np.uint16)
                for y in range(480):
                    depth_value = int(500 + y * 3)  # Depth increases with y (ground plane)
                    depth_frame[y, :] = depth_value
                
                # Add simulated obstacle
                cv2.rectangle(depth_frame, (300, 200), (350, 280), 1500, -1)
            
            # Convert RGB to BGR for consistency
            rgb_display = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Analyze depth map
            depth_stats = depth_debugger.analyze_depth_map(depth_frame)
            raw_viz, filtered_viz = depth_debugger.visualize_depth(depth_frame, depth_stats)
            
            # Detect AprilTags
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            tag_detections = tag_tester.detect_with_diagnostics(gray, K)
            
            # Detect obstacles
            obstacles = obstacle_detector.detect_obstacles(depth_frame, K)
            
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
            
            # Save debug images periodically (instead of displaying)
            if frame_count % 60 == 0:
                try:
                    debug_view = create_debug_visualization(
                        rgb_display, filtered_viz, tag_detections, obstacles, depth_stats, current_mode
                    )
                    cv2.imwrite(f"/workspace/debug_frame_{frame_count:04d}.png", debug_view)
                    print(f"\n📸 Saved debug image: debug_frame_{frame_count:04d}.png")
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
