"""
OAK-D Lite AprilTag Detection Module
Detects AprilTags and estimates their pose relative to the camera
"""

import cv2
import numpy as np
from apriltag import Detector
from dataclasses import dataclass
from typing import Optional, List
import depthai as dai


@dataclass
class AprilTagDetection:
    """Represents a detected AprilTag with pose information"""
    tag_id: int
    tag_family: str
    center: tuple  # (u, v) in pixel coordinates
    pose: np.ndarray  # 4x4 transformation matrix (camera to tag)
    distance: float  # Distance from camera in meters
    bearing: float  # Angle relative to camera center in radians
    confidence: float


class AprilTagDetector:
    """
    AprilTag detector optimized for ground-mounted tags
    Uses Oak-D Lite stereo depth for initial depth estimation
    """
    
    def __init__(self, tag_family: str = "tag36h11", 
                 quad_decimate: float = 2.0,
                 quad_sigma: float = 0.0):
        """
        Initialize AprilTag detector
        
        Args:
            tag_family: AprilTag family (e.g., "tag36h11", "tag16h5")
            quad_decimate: Detection resolution (higher = faster but less accurate)
            quad_sigma: Gaussian blur sigma for detection
        """
        self.detector = Detector(
            families=tag_family,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            nthreads=4
        )
        self.tag_family = tag_family
        
        # Camera intrinsics (will be updated from OAK-D)
        self.fx = 800.0  # Approximate focal length
        self.fy = 800.0
        self.cx = 640.0  # Principal point
        self.cy = 360.0
        
        # Tag size in meters (should be configured based on actual tags)
        self.tag_size = 0.16  # 16cm standard AprilTag
        
    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Set camera intrinsics from OAK-D calibration"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
    def detect_tags(self, image_gray: np.ndarray, 
                    depth_map: Optional[np.ndarray] = None) -> List[AprilTagDetection]:
        """
        Detect AprilTags in grayscale image
        
        Args:
            image_gray: Grayscale input image
            depth_map: Optional depth map from OAK-D for improved pose estimation
            
        Returns:
            List of AprilTagDetection objects
        """
        detections = []
        
        # Run AprilTag detection
        results = self.detector.detect(image_gray)
        
        for result in results:
            # Get tag center in pixel coordinates
            corners = result.corners
            center = (int(np.mean(corners[:, 0])), int(np.mean(corners[:, 1])))
            
            # Estimate pose using PnP if we have good corner detection
            if len(corners) == 4:
                # Object points (3D tag corners in tag coordinate system)
                obj_points = np.array([
                    [-self.tag_size/2, -self.tag_size/2, 0],
                    [self.tag_size/2, -self.tag_size/2, 0],
                    [self.tag_size/2, self.tag_size/2, 0],
                    [-self.tag_size/2, self.tag_size/2, 0]
                ], dtype=np.float32)
                
                # Image points (2D detected corners)
                img_points = corners.astype(np.float32)
                
                # Camera matrix
                K = np.array([
                    [self.fx, 0, self.cx],
                    [0, self.fy, self.cy],
                    [0, 0, 1]
                ])
                
                # Distortion coefficients (assuming rectified image)
                dist_coeffs = np.zeros(5)
                
                # Solve PnP
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, K, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE  # Optimized for planar targets
                )
                
                if success:
                    # Convert rotation vector to matrix
                    R, _ = cv2.Rodrigues(rvec)
                    
                    # Build 4x4 transformation matrix (camera to tag)
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = tvec.flatten()
                    
                    # Calculate distance and bearing
                    distance = np.linalg.norm(tvec)
                    bearing = np.arctan2(tvec[0], tvec[2])  # Horizontal angle
                    
                    detection = AprilTagDetection(
                        tag_id=result.tag_id,
                        tag_family=self.tag_family,
                        center=center,
                        pose=pose,
                        distance=distance,
                        bearing=bearing,
                        confidence=result.decision_margin
                    )
                    detections.append(detection)
        
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
        """Configure OAK-D Lite stereo depth + RGB pipeline"""
        self.pipeline = dai.Pipeline()
        
        # Define sources and outputs
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        
        # RGB camera configuration
        cam_rgb.setPreviewSize(1280, 720)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(30)
        
        # Stereo depth configuration
        stereo.setConfidenceThreshold(200)
        stereo.setRectifyEdgeFillColor(0)  # Black fill for invalid pixels
        stereo.setOutputSize(1280, 720)
        
        # Link nodes
        cam_rgb.preview.link(stereo.inputLeft)  # Use RGB as left input for aligned depth
        
        # Create outputs
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")
        
        cam_rgb.preview.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)
        
        return self.pipeline
    
    def start(self):
        """Start OAK-D device and pipeline"""
        if self.pipeline is None:
            self.setup_oakd_pipeline()
        
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        # Get camera intrinsics from calibration
        calib = self.device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
        
        # Update detector with actual intrinsics
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]
        self.april_detector.set_camera_intrinsics(fx, fy, cx, cy)
        
        print(f"OAK-D initialized with intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
    def get_frame_data(self):
        """
        Get synchronized RGB and depth frames
        
        Returns:
            Tuple of (rgb_frame, depth_frame, timestamp) or (None, None, None) if unavailable
        """
        rgb_packet = self.q_rgb.tryGet()
        depth_packet = self.q_depth.tryGet()
        
        if rgb_packet is not None and depth_packet is not None:
            rgb_frame = rgb_packet.getCvFrame()
            depth_frame = depth_packet.getFrame()  # Depth in mm
            
            # Convert BGR to RGB if needed
            if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            
            return rgb_frame, depth_frame, rgb_packet.getTimestampDevice()
        
        return None, None, None
    
    def detect_tags_in_frame(self, rgb_frame: np.ndarray, 
                            depth_frame: np.ndarray) -> List[AprilTagDetection]:
        """
        Detect AprilTags in a single frame
        
        Args:
            rgb_frame: RGB image from OAK-D
            depth_frame: Depth map from OAK-D (in mm)
            
        Returns:
            List of AprilTagDetection objects
        """
        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        # Detect tags
        detections = self.april_detector.detect_tags(gray, depth_frame)
        
        # Filter for ground-level tags
        ground_tags = self.april_detector.filter_ground_tags(detections)
        
        return ground_tags
    
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
