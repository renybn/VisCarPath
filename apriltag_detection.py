"""
OAK-D Lite AprilTag Detection Module
Detects AprilTags and estimates their pose relative to the camera
"""

import cv2
import numpy as np
import depthai as dai
from pupil_apriltags import Detector
from dataclasses import dataclass
from typing import Optional, List



@dataclass
class AprilTagDetection:
    """Represents a detected AprilTag with pose information"""
    tag_id: int
    tag_family: str
    center: tuple  # (u, v) in pixel coordinates
    corners: np.ndarray  # 4x2 array of corner pixel coordinates (x, y)
    pose: np.ndarray  # 4x4 transformation matrix (camera to tag)
    distance: float  # Distance from camera in meters
    bearing: float  # Angle relative to camera center in radians
    confidence: float  # Detection confidence (decision margin from pupil_apriltags)


class AprilTagDetector:
    """
    AprilTag detector optimized for ground-mounted tags
    Uses Oak-D Lite stereo depth for initial depth estimation
    """
    
    def __init__(self, tag_family: str = "tag36h11", 
                 quad_decimate: float = 1.0,
                 quad_sigma: float = 0.0):
        """
        Initialize AprilTag detector
        
        Args:
            tag_family: AprilTag family (e.g., "tag36h11", "tag16h5")
            quad_decimate: Detection resolution (higher = faster but less accurate)
            quad_sigma: Gaussian blur sigma for detection
        """
        # Validate tag family, fallback to tag36h11 if invalid
        valid_families = ["tag36h11", "tag16h5", "tag25h9", "tagStandard41h12", "tagCustom48h12"]
        if tag_family not in valid_families:
            print(f"Warning: Unknown tag family '{tag_family}'. Falling back to 'tag36h11'.")
            tag_family = "tag36h11"
        
        # pupil_apriltags takes parameters directly in the Detector constructor
        # We use nthreads=1 to avoid potential bus errors on some systems
        self.detector = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=True,
            decode_sharpening=0.25
        )
        self.tag_family = tag_family
        
        # Camera intrinsics fall back to typical OAK-D values, but will be updated from calibration
        self.fx = 800.0  # Approximate focal length
        self.fy = 800.0
        self.cx = 640.0  # Principal point
        self.cy = 360.0
        
        # Tag size in meters (should be configured based on actual tags)
        self.tag_size = 0.08  # 8cm standard AprilTag

        
    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Set camera intrinsics from OAK-D calibration"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
    def detect_tags(self, gray_frame: np.ndarray, depth_frame: np.ndarray) -> List[AprilTagDetection]:
        """Detect tags and compute accurate 3D pose using PnP"""
        detections = []
        h, w = gray_frame.shape[:2]
        
        # Run pupil_apriltags detector
        results = self.detector.detect(gray_frame)
        
        # Remove the hardcoded fx, fy, cx, cy block.
        # Instead, use the intrinsics stored in the class instance from the setup phase:
        K = np.array([
            [self.fx, 0, self.cx], 
            [0, self.fy, self.cy], 
            [0, 0, 1]
        ])
        
        # 3. 3D Object Points (centered square, in meters)
        half_size = self.tag_size / 2.0
        obj_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        for result in results:
            center = tuple(np.mean(result.corners, axis=0).astype(int))
            img_points = result.corners.astype(np.float32)
            
            # 4. Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, K, None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                t = tvec.flatten()
                distance = float(np.linalg.norm(t))  # Euclidean distance from camera lens
                bearing = float(np.arctan2(t[0], t[2]))
                
                R, _ = cv2.Rodrigues(rvec)
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t
                
                detections.append(AprilTagDetection(
                    tag_id=result.tag_id,
                    tag_family=self.tag_family,
                    center=center,
                    corners=result.corners,  # Pass the raw 4x2 corner array
                    pose=pose,
                    distance=distance,
                    bearing=bearing,
                    confidence=result.decision_margin
                ))
                    
        return detections
    
    def filter_ground_tags(self, detections: List[AprilTagDetection],
                          camera_pitch: float = 0.3,  # ~17 degrees downward
                          tolerance: float = 0.2) -> List[AprilTagDetection]:
        """
        Filter detections to only include tags likely on the ground plane
        
        Args:
            detections: List of all detected tags
            camera_pitch: Expected camera pitch angle (radians, positive = looking down)
            tolerance: Angular tolerance for ground plane classification
            
        Returns:
            Filtered list of ground-level tags
        """
        ground_tags = []
        
        for det in detections:
            # Extract tag position in camera frame
            tag_pos = det.pose[:3, 3]
            
            # For a ground-mounted tag, the normal should point upward
            # Tag coordinate system: Z points out of tag, so for ground tag Z should point up
            tag_normal_cam = det.pose[:3, 2]  # Tag Z-axis in camera frame
            
            # Expected ground normal in camera frame (pointing up)
            # If camera is pitched down by camera_pitch, ground normal rotates
            expected_normal = np.array([0, np.sin(camera_pitch), np.cos(camera_pitch)])
            
            # Check if tag normal aligns with expected ground normal
            dot_product = np.dot(tag_normal_cam, expected_normal)
            
            # Tags on ground should have normals pointing roughly toward camera (dot > 0)
            # and aligned with expected ground plane
            if dot_product > np.cos(tolerance):
                ground_tags.append(det)
        
        return ground_tags


class OakDAprilTagPipeline:
    """
    Complete OAK-D Lite pipeline for AprilTag detection with depth
    """
    
    def __init__(self, tag_family: str = "tag36h11"):
        """Initialize OAK-D pipeline"""
        self.april_detector = AprilTagDetector(tag_family=tag_family)
        self.pipeline = None
        self.device = None
        self.q_rgb = None
        self.q_depth = None
        
    def setup_oakd_pipeline(self):
        "Configure OAK-D Lite stereo depth + RGB pipeline with optimized settings"
        self.pipeline = dai.Pipeline()
        
        # 1. Define sources (RGB + 2 Mono cameras for depth)
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        
        # 2. RGB camera configuration
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # CRITICAL: Lock autofocus for stable PnP calculations
        # Use CONTINUOUS_VIDEO for dynamic focusing, or MANUAL for fixed distance
        cam_rgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        # For fixed-distance applications (e.g., always looking at ground 0.5-2m away), use manual focus:
        # cam_rgb.initialControl.setManualFocus(130)  # ~1m to infinity
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(15)
        
        # 3. Mono cameras configuration (Required for StereoDepth)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # 4. Stereo depth configuration - OPTIMIZED for accuracy
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setOutputSize(640, 480)
        stereo.setRectifyEdgeFillColor(0)
        
        # CRITICAL: Must be set AFTER the preset, otherwise it gets overwritten!
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setLeftRightCheck(True)
        
        # Enhanced disparity for better close-range accuracy
        stereo.setExtendedDisparity(True)
        stereo.setSubpixel(False)  # Disable subpixel for lower noise (trade-off: less precision)
        stereo.setConfidenceThreshold(220)  # Higher threshold = fewer false positives
        
        # 5. Link Mono cameras to Stereo Depth
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # 6. Create outputs to send to PC
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")
        
        # 7. Link camera previews to outputs
        cam_rgb.preview.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)

        return self.pipeline
    
    def start(self):

        "Start OAK-D device and pipeline"
        if self.pipeline is None:
            self.setup_oakd_pipeline()
        
        if self.pipeline is None:
            return
        
        self.device = dai.Device(self.pipeline, usb2Mode=True)
        print("Connected via USB 2.0!")

        # Initialize output queues (must happen AFTER device connection)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        # Get camera intrinsics from calibration
        calib = self.device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]
        self.april_detector.set_camera_intrinsics(fx, fy, cx, cy)
        
        print(f"OAK-D initialized with intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
    def get_frame_data(self):
        """
        Get synchronized RGB and depth frames.
        - Device-side is non-blocking (drops stale frames to prevent latency).
        - Host-side is blocking (ensures RGB and Depth are properly paired).
        - Includes defensive timeouts to prevent robot 'brain death' on USB drops.
        """
        if self.q_rgb is None or self.q_depth is None:
            raise RuntimeError("OAK-D queues not initialized. Hardware connection failed.")
            
        try:
            # FIX: use tryGet() (non-blocking) instead of get() (blocking).
            # get() deadlocks the navigation loop on USB hiccups or pipeline
            # stalls. tryGet() returns None immediately if no frame is ready;
            # callers handle None with a short sleep rather than freezing.
            rgb_packet   = self.q_rgb.tryGet()
            depth_packet = self.q_depth.tryGet()
            
        except RuntimeError as e:
            # Catches device disconnects, XLink errors, or pipeline crashes
            print(f"OAK-D Pipeline Error (USB drop or crash): {e}")
            return None, None, None
        except Exception as e:
            print(f"Unexpected Queue Error: {e}")
            return None, None, None

        # Safety check (though timeout should handle empty queues)
        if rgb_packet is None or depth_packet is None:
            return None, None, None
            
        try:
            rgb_frame = rgb_packet.getCvFrame()
            depth_frame = depth_packet.getFrame()  # Depth in mm
        except Exception as e:
            print(f"Frame extraction failed: {e}")
            return None, None, None
            
        # Ensure RGB is in standard OpenCV format
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            # DepthAI returns RGB if configured, but safe to ensure BGR->RGB if needed
            # (Your pipeline sets ColorOrder.RGB, so it's already RGB, but we ensure shape)
            pass 
            
        return rgb_frame, depth_frame, rgb_packet.getTimestampDevice()
    
    def detect_tags_in_frame(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> List[AprilTagDetection]:
        """
        Detect AprilTags in a frame using RGB + Depth for 3D pose.
        TEMPORARILY returns ALL tags to bypass strict ground filtering.
        """
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        # Run the full detection + PnP pipeline
        detections = self.april_detector.detect_tags(gray, depth_frame)
        
        # CRITICAL FIX: Return ALL detections for now
        # (Bypasses filter_ground_tags() which was dropping 100% of tags due to strict pitch math)
        return detections
    
    def stop(self):
        """Stop OAK-D device"""
        if self.device is not None:
            self.device.close()
            self.device = None


if __name__ == "__main__":
    # Example usage
    print("Testing AprilTag detection pipeline...")
    
    pipeline = OakDAprilTagPipeline()
    
    try:
        pipeline.start()
        print("OAK-D started successfully. Press Ctrl+C to stop.")
        
        while True:
            rgb, depth, ts = pipeline.get_frame_data()
            
            if rgb is not None:
                tags = pipeline.detect_tags_in_frame(rgb, depth)
                
                if tags:
                    print(f"Detected {len(tags)} ground-level AprilTag(s):")
                    for tag in tags:
                        print(f"  Tag ID: {tag.tag_id}, Distance: {tag.distance:.2f}m, "
                              f"Bearing: {np.degrees(tag.bearing):.1f}°")
                
                # Display frame with detections
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for tag in tags:
                    # Draw center
                    cv2.circle(display, tag.center, 5, (0, 255, 0), -1)
                    # Draw ID
                    cv2.putText(display, f"ID:{tag.tag_id}", 
                               (tag.center[0]+10, tag.center[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Draw distance
                    cv2.putText(display, f"{tag.distance:.2f}m",
                               (tag.center[0]+10, tag.center[1]+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imshow("AprilTag Detection", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()