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
                 max_iterations: int = 1000):
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
        """
        Detect ground plane from depth map using RANSAC
        
        Args:
            depth_map: Depth map in meters (or mm, will be converted)
            camera_intrinsics: 3x3 camera intrinsic matrix
            tag_masks: Optional binary mask marking AprilTag regions to exclude
            
        Returns:
            GroundPlane object or None if detection fails
        """
        h, w = depth_map.shape
        
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
        
        # RANSAC plane fitting
        best_normal = None
        best_distance = None
        best_inliers = 0
        best_inlier_mask = None
        
        for _ in range(self.max_iterations):
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
                best_inlier_mask = inlier_mask
        
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
        h, w = depth_map.shape
        
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
                
                # Calculate severity based on height and proximity
                avg_height = np.mean(height_above_ground[labels == i])
                severity = min(1.0, avg_height / 0.5) * (1.0 / distance)
                
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
    Creates navigable paths avoiding obstacles toward AprilTag targets
    """
    
    def __init__(self,
                 robot_width: float = 0.5,  # Robot width in meters
                 min_clearance: float = 0.2,  # Minimum clearance from obstacles
                 max_lookahead: float = 3.0):  # Maximum lookahead distance
        """
        Initialize path planner
        
        Args:
            robot_width: Robot width including safety margin
            min_clearance: Minimum distance from obstacles
            max_lookahead: Maximum planning horizon
        """
        self.robot_width = robot_width
        self.min_clearance = min_clearance
        self.max_lookahead = max_lookahead
        
    def create_cost_map(self, width: int, height: int,
                       obstacles: List[Obstacle],
                       camera_intrinsics: np.ndarray,
                       ground_plane: GroundPlane) -> np.ndarray:
        """
        Create 2D cost map for path planning
        
        Args:
            width, height: Map dimensions
            obstacles: List of detected obstacles
            camera_intrinsics: Camera intrinsics
            ground_plane: Ground plane parameters
            
        Returns:
            2D cost map (lower cost = more navigable)
        """
        # Initialize cost map
        cost_map = np.ones((height, width), dtype=np.float32)
        
        # Add obstacle costs
        for obs in obstacles:
            if obs.distance > self.max_lookahead:
                continue
                
            # Project obstacle influence radius
            influence_radius = int((self.robot_width/2 + self.min_clearance) * 
                                   camera_intrinsics[0, 0] / obs.distance)
            
            # Draw cost gradient around obstacle
            y, x = np.ogrid[:height, :width]
            dist_from_obs = np.sqrt((x - obs.center[0])**2 + **(y - obs.center[1])2)
            
            # Cost increases near obstacle
            obstacle_cost = np.exp(-dist_from_obs / (influence_radius / 2))
            cost_map = np.maximum(cost_map, obstacle_cost * 10)  # High cost near obstacles
        
        # Prefer center of image (straight ahead)
        center_x = width // 2
        y, x = np.ogrid[:height, :width]
        center_preference = np.abs(x - center_x) / width
        cost_map += center_preference
        
        return cost_map
    
    def plan_path_to_tag(self, tag_detection,
                        obstacles: List[Obstacle],
                        ground_plane: GroundPlane,
                        camera_intrinsics: np.ndarray,
                        image_shape: tuple) -> Optional[List[PathSegment]]:
        """
        Plan path from current position to AprilTag target
        
        Args:
            tag_detection: Target AprilTag detection
            obstacles: List of obstacles to avoid
            ground_plane: Ground plane parameters
            camera_intrinsics: Camera intrinsics
            image_shape: Image dimensions (h, w)
            
        Returns:
            List of PathSegment objects or None if no path found
        """
        h, w = image_shape
        
        # Create cost map
        cost_map = self.create_cost_map(
            w, h, obstacles, camera_intrinsics, ground_plane
        )
        
        # Start position (bottom center of image - robot position)
        start_u, start_v = w // 2, int(h * 0.8)
        
        # Goal position (tag center)
        goal_u, goal_v = tag_detection.center
        
        # Tag 3D position
        tag_pos = tag_detection.pose[:3, 3]
        
        # Simple A* or gradient descent path planning
        # For now, use simple straight-line with obstacle avoidance
        
        path_segments = []
        
        # Direct path to tag
        direct_vector = tag_pos - np.array([0, 0, 0])  # From camera origin
        direct_distance = np.linalg.norm(direct_vector)
        
        if direct_distance > self.max_lookahead:
            # Scale to max lookahead
            direct_vector = direct_vector * (self.max_lookahead / direct_distance)
        
        # Check for obstacles along direct path
        path_clear = True
        min_clearance = float('inf')
        
        for obs in obstacles:
            # Calculate perpendicular distance from obstacle to path
            path_direction = direct_vector / np.linalg.norm(direct_vector)
            obs_vector = obs.position_3d
            proj_length = np.dot(obs_vector, path_direction)
            closest_point = path_direction * proj_length
            perp_distance = np.linalg.norm(obs_vector - closest_point)
            
            if 0 < proj_length < np.linalg.norm(direct_vector):
                min_clearance = min(min_clearance, perp_distance)
                
                if perp_distance < (self.robot_width/2 + self.min_clearance):
                    path_clear = False
        
        if path_clear:
            # Direct path is clear
            segment = PathSegment(
                start=np.array([0, 0, 0]),
                end=direct_vector,
                width=self.robot_width,
                clearance=min_clearance,
                cost=direct_distance
            )
            path_segments.append(segment)
        else:
            # Need to route around obstacles
            # Simple workaround: find gap between obstacles
            path_segments = self._find_alternative_path(
                tag_pos, obstacles, cost_map, start_u, start_v, goal_u, goal_v
            )
        
        return path_segments if path_segments else None
    
    def _find_alternative_path(self, goal_pos: np.ndarray,
                              obstacles: List[Obstacle],
                              cost_map: np.ndarray,
                              start_u: int, start_v: int,
                              goal_u: int, goal_v: int) -> Optional[List[PathSegment]]:
        """Find alternative path when direct path is blocked"""
        # Simplified: just return None to indicate no path found
        # In full implementation, would use A* or RRT
        return None


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
        """Create binary mask marking tag regions"""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for tag in tag_detections:
            # Could draw tag bounding boxes here
            # For now, just return empty mask
            pass
        
        return mask
    
    def process_frame(self, depth_map: np.ndarray,
                     tag_detections: list,
                     image_shape: tuple) -> dict:
        """
        Process single frame to detect ground, obstacles, and plan path
        
        Args:
            depth_map: Depth map from OAK-D (in mm)
            tag_detections: List of AprilTagDetection objects
            image_shape: Image dimensions (h, w)
            
        Returns:
            Dictionary with ground_plane, obstacles, and path information
        """
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not set")
        
        # Create tag mask
        tag_mask = self.create_tag_mask(image_shape, tag_detections)
        
        # Detect ground plane
        ground_plane = self.ground_detector.detect_ground_plane(
            depth_map, self.camera_intrinsics, tag_mask
        )
        
        # Detect obstacles
        obstacles = []
        if ground_plane is not None:
            obstacles = self.obstacle_detector.detect_obstacles(
                depth_map, ground_plane, self.camera_intrinsics, tag_mask
            )
        
        # Plan path to primary tag (closest one)
        path = None
        if tag_detections and ground_plane:
            primary_tag = min(tag_detections, key=lambda t: t.distance)
            path = self.path_planner.plan_path_to_tag(
                primary_tag, obstacles, ground_plane,
                self.camera_intrinsics, image_shape
            )
        
        return {
            'ground_plane': ground_plane,
            'obstacles': obstacles,
            'path': path,
            'tag_mask': tag_mask
        }


if __name__ == "__main__":
    print("Ground plane and obstacle detection module loaded.")
    print("This module should be used with apriltag_detection.py")
