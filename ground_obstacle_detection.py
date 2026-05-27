"""
Ground Plane Detection and Obstacle Avoidance Module
Detects ground plane, identifies obstacles, and creates navigable path maps
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class TerrainType(Enum):
    """Classification of terrain types"""
    GROUND = 0
    OBSTACLE = 1
    UNKNOWN = 2
    TAG = 3


@dataclass
class GroundPlane:
    """Represents detected ground plane parameters"""
    normal: np.ndarray  # 3D normal vector in camera frame
    distance: float  # Distance from camera to plane along normal
    confidence: float  # Confidence score [0, 1]
    inliers: int  # Number of inlier points
    bounds: tuple  # (min_x, max_x, min_y, max_y) in image coordinates


@dataclass
class Obstacle:
    """Represents a detected obstacle"""
    center: tuple  # (u, v) in pixel coordinates
    position_3d: np.ndarray  # 3D position in camera frame (meters)
    size: tuple  # (width, height) in pixels
    distance: float  # Distance from camera in meters
    bearing: float  # Horizontal angle in radians
    severity: float  # Obstacle severity [0, 1]


@dataclass
class PathSegment:
    """Represents a segment of the planned path"""
    start: np.ndarray  # 3D start position (camera frame)
    end: np.ndarray  # 3D end position (camera frame)
    width: float  # Available width in meters
    clearance: float  # Minimum clearance from obstacles in meters
    cost: float  # Path cost (lower is better)


class GroundPlaneDetector:
    """
    Detects ground plane using RANSAC on depth data
    Optimized for downward-tilted camera viewing ground with AprilTags
    """
    
    def __init__(self, 
                 ransac_threshold: float = 0.05,  # 5cm threshold
                 min_inliers: int = 100,
                 max_iterations: int = 100):
        """
        Initialize ground plane detector
        
        Args:
            ransac_threshold: Maximum distance for point to be considered inlier (meters)
            min_inliers: Minimum number of inliers to accept plane detection
            max_iterations: RANSAC iterations
        """
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations
        
        # Expected camera pitch (radians, positive = looking down)
        self.expected_pitch = 0.3  # ~17 degrees
        
    def detect_ground_plane(self, depth_map: np.ndarray, 
                           camera_intrinsics: np.ndarray,
                           tag_masks: Optional[np.ndarray] = None) -> Optional[GroundPlane]:
        """Detect ground plane from depth map using RANSAC"""
        
        # 🔥 CRITICAL: Downsample to 320x240 for RANSAC (cuts pixels by 75%)
        h, w = depth_map.shape
        if h > 240 or w > 320:
            depth_map = cv2.resize(depth_map, (320, 240), interpolation=cv2.INTER_NEAREST)
            # Also resize tag_mask if provided
            if tag_masks is not None:
                tag_masks = cv2.resize(tag_masks, (320, 240), interpolation=cv2.INTER_NEAREST)
            # Scale intrinsics for new resolution
            scale_x = 320 / w
            scale_y = 240 / h
            camera_intrinsics = camera_intrinsics.copy()
            camera_intrinsics[0, 0] *= scale_x  # fx
            camera_intrinsics[1, 1] *= scale_y  # fy
            camera_intrinsics[0, 2] *= scale_x  # cx
            camera_intrinsics[1, 2] *= scale_y  # cy
        
        h, w = depth_map.shape  # Update for new size
        
        # Convert depth to meters if needed (assume >1000 means mm)
        if np.median(depth_map[depth_map > 0]) > 100:
            depth_meters = depth_map.astype(np.float32) / 1000.0
        else:
            depth_meters = depth_map.astype(np.float32)
        
        # Create point cloud
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Mask valid depth values
        valid_mask = (depth_meters > 0.1) & (depth_meters < 10.0)  # 10cm to 10m range
        
        # Exclude tag regions if provided
        if tag_masks is not None:
            valid_mask = valid_mask & (tag_masks == 0)
        
        # Exclude bottom edge (likely robot body)
        valid_mask[int(h * 0.9):, :] = False
        
        # Get valid point indices
        valid_indices = np.where(valid_mask)
        
        if len(valid_indices[0]) < self.min_inliers:
            return None
        
        # Convert to 3D points in camera frame
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        x_cam = (x_coords[valid_mask] - cx) * depth_meters[valid_mask] / fx
        y_cam = (y_coords[valid_mask] - cy) * depth_meters[valid_mask] / fy
        z_cam = depth_meters[valid_mask]
        
        points_3d = np.column_stack([x_cam, y_cam, z_cam])
        
        # RANSAC plane fitting (original structure, but with downsampling applied above)
        best_normal = None
        best_distance = None
        best_inliers = 0
        best_inlier_mask = None  # ← MUST track this for bounds calculation later
        
        for _ in range(self.max_iterations):  # Now only 100 iterations instead of 1000
            # Sample 3 random points
            if len(points_3d) < 3:
                break
                
            idx = np.random.choice(len(points_3d), 3, replace=False)
            sample = points_3d[idx]
            
            # Compute plane from 3 points
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
                
            normal = normal / norm
            
            # Plane equation: n·x + d = 0
            distance = -np.dot(normal, sample[0])
            
            # Count inliers
            distances = np.abs(np.dot(points_3d, normal) + distance)
            inlier_mask = distances < self.ransac_threshold
            num_inliers = np.sum(inlier_mask)
            
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_normal = normal
                best_distance = distance
                best_inlier_mask = inlier_mask  # ← CRITICAL: Keep tracking this!

        # Validate detection
        if best_inliers < self.min_inliers:
            return None
        
        # Check if normal is consistent with expected ground orientation
        # Ground normal should point upward (positive Y in camera frame when pitched down)
        expected_normal = np.array([0, np.sin(self.expected_pitch), np.cos(self.expected_pitch)])
        
        # Ensure normal points toward camera (for ground below)
        if np.dot(best_normal, expected_normal) < 0:
            best_normal = -best_normal
            best_distance = -best_distance
        
        # Calculate confidence based on inlier ratio
        confidence = min(1.0, best_inliers / len(points_3d))
        
        # Get bounds of inlier region in image
        inlier_coords_y = valid_indices[0][best_inlier_mask]
        inlier_coords_x = valid_indices[1][best_inlier_mask]
        
        if len(inlier_coords_x) > 0:
            bounds = (
                int(np.min(inlier_coords_x)),
                int(np.max(inlier_coords_x)),
                int(np.min(inlier_coords_y)),
                int(np.max(inlier_coords_y))
            )
        else:
            bounds = (0, w, 0, h)
        
        return GroundPlane(
            normal=best_normal,
            distance=best_distance,
            confidence=confidence,
            inliers=best_inliers,
            bounds=bounds
        )


class ObstacleDetector:
    """
    Detects obstacles by analyzing deviations from ground plane
    """
    
    def __init__(self,
                 height_threshold: float = 0.05,  # 5cm above ground
                 min_obstacle_size: int = 50,  # Minimum pixels
                 max_obstacle_distance: float = 5.0):  # Maximum detection range
        """
        Initialize obstacle detector
        
        Args:
            height_threshold: Minimum height above ground to be considered obstacle (meters)
            min_obstacle_size: Minimum connected component size (pixels)
            max_obstacle_distance: Maximum detection range (meters)
        """
        self.height_threshold = height_threshold
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_distance = max_obstacle_distance
        
    def detect_obstacles(self, depth_map: np.ndarray,
                        ground_plane: GroundPlane,
                        camera_intrinsics: np.ndarray,
                        tag_masks: Optional[np.ndarray] = None) -> List[Obstacle]:
        """
        Detect obstacles as points significantly above ground plane
        
        Args:
            depth_map: Depth map in mm
            ground_plane: Detected ground plane
            camera_intrinsics: Camera intrinsic matrix
            tag_masks: Binary mask of tag regions to exclude
            
        Returns:
            List of detected obstacles
        """
        # 🔥 Downsample for obstacle detection too
        orig_h, orig_w = depth_map.shape
        if orig_h > 240 or orig_w > 320:
            depth_map = cv2.resize(depth_map, (320, 240), interpolation=cv2.INTER_NEAREST)
            if tag_masks is not None:
                tag_masks = cv2.resize(tag_masks, (320, 240), interpolation=cv2.INTER_NEAREST)
            # Scale intrinsics for new resolution
            scale_x = 320 / orig_w
            scale_y = 240 / orig_h
            camera_intrinsics = camera_intrinsics.copy()
            camera_intrinsics[0, 0] *= scale_x
            camera_intrinsics[1, 1] *= scale_y
            camera_intrinsics[0, 2] *= scale_x
            camera_intrinsics[1, 2] *= scale_y
        
        h, w = depth_map.shape  # Update for new size
        
        # Convert depth to meters
        if np.median(depth_map[depth_map > 0]) > 100:
            depth_meters = depth_map.astype(np.float32) / 1000.0
        else:
            depth_meters = depth_map.astype(np.float32)
        
        # Create point cloud
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Valid depth range
        valid_mask = (depth_meters > 0.1) & (depth_meters <= self.max_obstacle_distance)
        
        if tag_masks is not None:
            valid_mask = valid_mask & (tag_masks == 0)
        
        # Get 3D points
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        x_cam = (x_coords - cx) * depth_meters / fx
        y_cam = (y_coords - cy) * depth_meters / fy
        z_cam = depth_meters
        
        # Calculate expected ground height at each point
        # Ground plane: n·p + d = 0, solve for y (height)
        n = ground_plane.normal
        d = ground_plane.distance
        
        # For each point, calculate height above ground
        # Point is on ground if: n_x*x + n_y*y + n_z*z + d = 0
        # Solve for y: y = -(n_x*x + n_z*z + d) / n_y
        if abs(n[1]) < 1e-6:
            return []  # Can't compute height
        
        expected_y = -(n[0] * x_cam + n[2] * z_cam + d) / n[1]
        actual_y = y_cam
        
        # Height above ground
        height_above_ground = actual_y - expected_y
        
        # Obstacle mask: points significantly above ground
        obstacle_mask = (height_above_ground > self.height_threshold) & valid_mask
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        obstacle_mask_uint8 = (obstacle_mask * 255).astype(np.uint8)
        obstacle_mask_cleaned = cv2.morphologyEx(
            obstacle_mask_uint8, cv2.MORPH_CLOSE, kernel
        )
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            obstacle_mask_cleaned, connectivity=8
        )
        
        obstacles = []
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= self.min_obstacle_size:
                # Get centroid
                centroid_u, centroid_v = centroids[i]
                
                # Get 3D position at centroid
                z_centroid = depth_meters[int(centroid_v), int(centroid_u)]
                x_centroid = (centroid_u - cx) * z_centroid / fx
                y_centroid = (centroid_v - cy) * z_centroid / fy
                
                position_3d = np.array([x_centroid, y_centroid, z_centroid])
                
                # Calculate distance and bearing
                distance = np.linalg.norm(position_3d)
                bearing = np.arctan2(x_centroid, z_centroid)
                
                # Calculate severity based on obstacle height above ground and proximity
                # Use the actual height_above_ground at the centroid, not global average
                centroid_height = height_above_ground[int(centroid_v), int(centroid_u)]
                severity = min(1.0, centroid_height / 0.3) * (1.0 / max(0.1, distance))
                
                obstacle = Obstacle(
                    center=(int(centroid_u), int(centroid_v)),
                    position_3d=position_3d,
                    size=(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                    distance=distance,
                    bearing=bearing,
                    severity=severity
                )
                obstacles.append(obstacle)
        
        # Sort by distance (closest first)
        obstacles.sort(key=lambda o: o.distance)
        
        return obstacles


class PathPlanner:
    """
    Reactive Vector Field Path Planner (Vectorized)
    Evaluates high-resolution heading candidates using NumPy broadcasting
    to balance target alignment and obstacle avoidance.
    """
    def __init__(self,
                 robot_width: float = 0.5,
                 min_clearance: float = 0.2,
                 max_lookahead: float = 3.0):
        self.robot_width = robot_width
        self.min_clearance = min_clearance
        self.max_lookahead = max_lookahead
        
        # High-resolution candidate angles: -40 to +40 degrees (1-degree steps)
        self.thetas = np.deg2rad(np.arange(-40, 41, 1))
        
        # Cost function weights
        self.w_target = 1.0
        self.w_obs = 5.0
        self.margin_rad = np.deg2rad(5)

    def plan_path_to_tag(self, tag_detection, obstacles: list,
                         ground_plane, camera_intrinsics: np.ndarray,
                         image_shape: tuple) -> Optional[list]:
        """
        Plan path using vectorized angular evaluation.
        """
        # 1. Extract target vector in X-Z plane (camera frame)
        tag_pos = tag_detection.pose[:3, 3]
        tag_x, tag_z = tag_pos[0], tag_pos[2]
        target_dist = np.hypot(tag_x, tag_z)
        
        if target_dist < 0.1:
            return None
            
        target_angle = np.arctan2(tag_x, tag_z)
        fx = camera_intrinsics[0, 0]
        
        # 2. Precompute obstacle angular blockers
        obs_blockers = []
        for obs in obstacles:
            obs_x, obs_z = obs.position_3d[0], obs.position_3d[2]
            d = np.hypot(obs_x, obs_z)
            
            if d < 0.1 or obs_z < 0:
                continue
                
            # Pinhole model: convert pixel width to real-world meters
            obs_width_m = (obs.size[0] * obs_z) / fx
            r_safe = (obs_width_m / 2.0) + (self.robot_width / 2.0) + self.min_clearance
            
            obs_angle = np.arctan2(obs_x, obs_z)
            
            if d <= r_safe:
                obs_blockers.append((obs_angle, np.pi, d))
            else:
                delta_theta = np.arcsin(np.clip(r_safe / d, -1.0, 1.0))
                obs_blockers.append((obs_angle, delta_theta, d))
                
        # 3. Vectorized Cost Evaluation
        # Target alignment cost: 0 (perfect alignment) to 2 (opposite direction)
        diff_target = self._wrap_angle(self.thetas - target_angle)
        cost_target = 1.0 - np.cos(diff_target)
        
        # Obstacle penalty (accumulated via broadcasting)
        cost_obs = np.zeros_like(self.thetas)
        for obs_angle, delta_theta, d in obs_blockers:
            diff_obs = self._wrap_angle(self.thetas - obs_angle)
            blocked_zone = delta_theta + self.margin_rad
            
            # Penetration depth: 1.0 at center of obstacle, 0.0 at edge of blocked zone
            penetration = np.maximum(0.0, 1.0 - (np.abs(diff_obs) / blocked_zone))
            proximity_factor = 1.0 / max(0.5, d)
            cost_obs += penetration * proximity_factor
            
        total_cost = (self.w_target * cost_target) + (self.w_obs * cost_obs)
        
        # 4. Select optimal heading
        best_idx = np.argmin(total_cost)
        best_angle = self.thetas[best_idx]
        
        # 5. Generate 3D Waypoint and PathSegment
        lookahead = min(self.max_lookahead, target_dist)
        wp_x = lookahead * np.sin(best_angle)
        wp_z = lookahead * np.cos(best_angle)
        wp_3d = np.array([wp_x, 0.0, wp_z])
        
        segment = PathSegment(
            start=np.array([0.0, 0.0, 0.0]),
            end=wp_3d,
            width=self.robot_width,
            clearance=self.min_clearance,
            cost=float(total_cost[best_idx])
        )
        
        return [segment]

    @staticmethod
    def _wrap_angle(angles: np.ndarray) -> np.ndarray:
        """Vectorized angle wrapping to [-pi, pi]"""
        return (angles + np.pi) % (2 * np.pi) - np.pi
    

class GroundAndObstaclePipeline:
    """
    Complete pipeline for ground detection, obstacle detection, and path planning
    """
    
    def __init__(self, robot_width: float = 0.5):
        """Initialize complete pipeline"""
        self.ground_detector = GroundPlaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.path_planner = PathPlanner(robot_width=robot_width)
        
        # Camera intrinsics (will be set from OAK-D)
        self.camera_intrinsics = None
        
    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Set camera intrinsics"""
        self.camera_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def create_tag_mask(self, image_shape: tuple, 
                       tag_detections: list) -> np.ndarray:
        """
        Create binary mask marking tag regions to exclude them from 
        ground plane estimation and obstacle detection.
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for tag in tag_detections:
            if hasattr(tag, 'corners') and tag.corners is not None:
                # pupil_apriltags returns corners as (4, 2) float array of (x, y)
                # OpenCV fillPoly requires int32 and shape (-1, 1, 2)
                pts = np.array(tag.corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
        
        # Dilate the mask to cover the tag's black border and stereo depth edge-noise
        if np.any(mask > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        return mask
    
    def process_frame(self, depth_map: np.ndarray,
                        tag_detections: list,
                        image_shape: tuple) -> dict:
            """
            Process single frame for ground and obstacle perception.
            Path planning is intentionally excluded to maintain separation of concerns.
            """
            if self.camera_intrinsics is None:
                raise ValueError("Camera intrinsics not set")
            
            print("[GROUND] Processing ground/obstacle perception...")
            tag_mask = self.create_tag_mask(image_shape, tag_detections)
            print(f"[GROUND]   - Created tag mask ({np.sum(tag_mask > 0)} pixels masked)")
            
            ground_plane = self.ground_detector.detect_ground_plane(
                depth_map, self.camera_intrinsics, tag_mask
            )
            
            obstacles = []
            if ground_plane is not None:
                obstacles = self.obstacle_detector.detect_obstacles(
                    depth_map, ground_plane, self.camera_intrinsics, tag_mask
                )
            else:
                print("[GROUND]   - Skipping obstacle detection (no ground plane)")
            
            result = {
                'ground_plane': ground_plane,
                'obstacles': obstacles,
                'tag_mask': tag_mask
            }
            print(f"[GROUND] Perception complete: {len(obstacles)} obstacles\n")
            return result


if __name__ == "__main__":
    print("Ground plane and obstacle detection module loaded.")
    print("This module should be used with apriltag_detection.py")
