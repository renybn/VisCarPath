"""
Main Autonomous Navigation Pipeline
Integrates OAK-D, AprilTag detection, ground/obstacle detection, 
Kalman filtering, and MPC control for real-time autonomous navigation
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import our modules
from apriltag_detection import OakDAprilTagPipeline, AprilTagDetection
from ground_obstacle_detection import (
    GroundAndObstaclePipeline, 
    GroundPlane, 
    Obstacle,
    PathSegment
)
from kalman_filter import ExtendedKalmanFilter, TagMeasurementFusion, VehicleState
from mpc_controller import PathFollowingController, MPCConfig


class NavigationState(Enum):
    """Navigation system states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    DETECTING_TAGS = "detecting_tags"
    PLANNING_PATH = "planning_path"
    NAVIGATING = "navigating"
    OBSTRUCTED = "obstructed"
    TARGET_REACHED = "target_reached"
    ERROR = "error"


@dataclass
class NavigationCommand:
    """Control command output from navigation system"""
    acceleration: float  # m/s²
    steering_rate: float  # rad/s
    target_velocity: float  # m/s
    timestamp: float  # Unix timestamp


class AutonomousNavigator:
    """
    Complete autonomous navigation system integrating all components
    
    Architecture:
    1. OAK-D Lite captures RGB + Depth frames
    2. AprilTag detector identifies ground-mounted tags
    3. Ground plane detector identifies navigable terrain
    4. Obstacle detector finds obstacles above ground plane
    5. Path planner creates collision-free path to target tag
    6. Kalman filter estimates vehicle state from tag observations
    7. MPC controller computes optimal control inputs
    """
    
    def __init__(self, 
                 robot_width: float = 0.5,
                 target_tag_id: Optional[int] = None):
        """
        Initialize complete navigation system
        
        Args:
            robot_width: Robot width in meters (for collision avoidance)
            target_tag_id: Specific tag ID to navigate to (None = closest visible)
        """
        print("Initializing Autonomous Navigation System...")
        
        # Initialize OAK-D pipeline
        self.oak_pipeline = OakDAprilTagPipeline()
        
        # Initialize ground/obstacle detection
        self.ground_pipeline = GroundAndObstaclePipeline(robot_width=robot_width)
        
        # Initialize Kalman filter
        initial_state = VehicleState(x=0, y=0, theta=0, v=0, omega=0)
        self.ekf = ExtendedKalmanFilter(
            initial_state=initial_state,
            process_noise=0.1,
            measurement_noise=0.5
        )
        self.tag_fusion = TagMeasurementFusion(self.ekf)
        
        # Initialize MPC controller
        mpc_config = MPCConfig(
            horizon=10,
            dt=0.1,
            max_velocity=1.5,
            max_acceleration=0.8,
            obstacle_safety_margin=0.3
        )
        self.mpc_controller = PathFollowingController(mpc_config)
        
        # System state
        self.navigation_state = NavigationState.IDLE
        self.current_command = NavigationCommand(0, 0, 0, time.time())
        
        # Target configuration
        self.target_tag_id = target_tag_id
        
        # Performance metrics
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        # Known tag map (can be pre-populated or learned online)
        self.tag_map: Dict[int, tuple] = {}
        
        print("Initialization complete.")
        
    def set_target_tag(self, tag_id: int):
        """Set specific target tag to navigate to"""
        self.target_tag_id = tag_id
        print(f"Target tag set to ID: {tag_id}")
        
    def add_known_tag(self, tag_id: int, x: float, y: float, theta: float = 0):
        """Add known tag position to map"""
        self.tag_map[tag_id] = (x, y, theta)
        self.tag_fusion.add_tag_to_map(tag_id, x, y, theta)
        
    def start(self):
        """Start the navigation system"""
        print("Starting OAK-D device...")
        self.oak_pipeline.start()
        
        # Set camera intrinsics for ground pipeline
        fx = self.oak_pipeline.april_detector.fx
        fy = self.oak_pipeline.april_detector.fy
        cx = self.oak_pipeline.april_detector.cx
        cy = self.oak_pipeline.april_detector.cy
        
        self.ground_pipeline.set_camera_intrinsics(fx, fy, cx, cy)
        
        self.navigation_state = NavigationState.DETECTING_TAGS
        print("Navigation system started.")
        
    def stop(self):
        """Stop the navigation system"""
        print("Stopping navigation system...")
        if self.oak_pipeline.device:
            self.oak_pipeline.stop()
        self.navigation_state = NavigationState.IDLE
        print("Navigation system stopped.")
        
    def process_frame(self) -> Optional[NavigationCommand]:
        """
        Process single frame and compute control command
        
        Returns:
            NavigationCommand or None if processing failed
        """
        frame_start = time.time()
        
        # Get frame data from OAK-D
        rgb_frame, depth_frame, timestamp = self.oak_pipeline.get_frame_data()
        
        if rgb_frame is None or depth_frame is None:
            return None
        
        h, w = rgb_frame.shape[:2]
        
        # Detect AprilTags
        tag_detections = self.oak_pipeline.detect_tags_in_frame(rgb_frame, depth_frame)
        
        # Learn new tag positions if not in map
        for tag in tag_detections:
            if tag.tag_id not in self.tag_map:
                # Estimate tag world position from current state
                state = self.ekf.get_state()
                tag_pos_cam = tag.pose[:3, 3]
                
                # Simple estimation: assume tag is at camera heading direction
                tag_world_x = state.x + tag_pos_cam[2] * np.cos(state.theta)
                tag_world_y = state.y + tag_pos_cam[2] * np.sin(state.theta)
                
                self.add_known_tag(tag.tag_id, tag_world_x, tag_world_y)
                print(f"Learned tag {tag.tag_id} at ({tag_world_x:.2f}, {tag_world_y:.2f})")
        
        # Select target tag
        target_tag = None
        if self.target_tag_id is not None:
            for tag in tag_detections:
                if tag.tag_id == self.target_tag_id:
                    target_tag = tag
                    break
        elif tag_detections:
            # Use closest tag as target
            target_tag = min(tag_detections, key=lambda t: t.distance)
        
        # Process ground and obstacles
        frame_data = self.ground_pipeline.process_frame(
            depth_frame, tag_detections, (h, w)
        )
        
        ground_plane = frame_data['ground_plane']
        obstacles = frame_data['obstacles']
        path = frame_data['path']
        
        # Update Kalman filter with tag measurements
        if tag_detections:
            self.tag_fusion.update_ekf_with_tags(tag_detections)
        
        # Predict next state
        self.ekf.predict(dt=0.1, control_input=(
            self.current_command.acceleration,
            self.current_command.steering_rate
        ))
        
        # Get current state estimate
        current_state = self.ekf.get_state()
        
        # Plan path if needed
        if target_tag and ground_plane and path is None:
            # Re-plan path to target
            path = self.ground_pipeline.path_planner.plan_path_to_tag(
                target_tag, obstacles, ground_plane,
                self.ground_pipeline.camera_intrinsics, (h, w)
            )
        
        # Determine navigation state
        if target_tag is None:
            self.navigation_state = NavigationState.DETECTING_TAGS
        elif obstacles and len(obstacles) > 0:
            closest_obstacle = min(obstacles, key=lambda o: o.distance)
            if closest_obstacle.distance < 1.0:
                self.navigation_state = NavigationState.OBSTRUCTED
            else:
                self.navigation_state = NavigationState.NAVIGATING
        elif target_tag and target_tag.distance < 0.5:
            self.navigation_state = NavigationState.TARGET_REACHED
        else:
            self.navigation_state = NavigationState.NAVIGATING
        
        # Compute control command
        if self.navigation_state in [NavigationState.NAVIGATING, NavigationState.PLANNING_PATH]:
            # Update MPC with current state
            state_array = np.array([
                current_state.x,
                current_state.y,
                current_state.theta,
                current_state.v,
                0.0  # Steering angle (not estimated)
            ])
            self.mpc_controller.update_state(state_array)
            
            # Convert path segments to format MPC expects
            if path:
                mpc_path = []
                for seg in path:
                    mpc_path.append(seg)
                
                # Compute control
                accel, steer = self.mpc_controller.compute_control(
                    mpc_path if mpc_path else [],
                    obstacles
                )
                
                # Limit commands based on state
                if self.navigation_state == NavigationState.OBSTRUCTED:
                    accel = max(-0.5, accel)  # Gentle braking
                
                self.current_command = NavigationCommand(
                    acceleration=accel,
                    steering_rate=steer,
                    target_velocity=current_state.v + accel * 0.1,
                    timestamp=time.time()
                )
            else:
                # No path available - stop
                self.current_command = NavigationCommand(0, 0, 0, time.time())
                
        elif self.navigation_state == NavigationState.TARGET_REACHED:
            # Stop at target
            self.current_command = NavigationCommand(0, 0, 0, time.time())
            
        elif self.navigation_state == NavigationState.OBSTRUCTED:
            # Slow down or stop
            self.current_command = NavigationCommand(
                acceleration=-0.3,
                steering_rate=0,
                target_velocity=max(0, current_state.v - 0.3),
                timestamp=time.time()
            )
        else:
            # Default: stop
            self.current_command = NavigationCommand(0, 0, 0, time.time())
        
        # Update FPS
        self.frame_count += 1
        if time.time() - self.last_frame_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_frame_time = time.time()
        
        return self.current_command
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get system diagnostics"""
        state = self.ekf.get_state()
        
        return {
            'navigation_state': self.navigation_state.value,
            'estimated_position': (state.x, state.y),
            'estimated_heading': np.degrees(state.theta),
            'estimated_velocity': state.v,
            'target_tag_id': self.target_tag_id,
            'known_tags': len(self.tag_map),
            'fps': self.fps,
            'control_acceleration': self.current_command.acceleration,
            'control_steering_rate': self.current_command.steering_rate
        }
    
    def run_visualization(self, show_diagnostics: bool = True):
        """
        Run navigation with visualization
        
        Args:
            show_diagnostics: Whether to display diagnostic overlay
        """
        print("Starting visualization mode. Press 'q' to quit.")
        
        try:
            while True:
                # Process frame
                command = self.process_frame()
                
                # Get latest frame for display
                rgb, depth, _ = self.oak_pipeline.get_frame_data()
                
                if rgb is None:
                    continue
                
                # Convert for display
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                h, w = display.shape[:2]
                
                # Get detections
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                tag_detections = self.oak_pipeline.april_detector.detect_tags(gray)
                ground_tags = self.oak_pipeline.april_detector.filter_ground_tags(tag_detections)
                
                # Draw tag detections
                for tag in ground_tags:
                    color = (0, 255, 0) if tag.tag_id == self.target_tag_id else (0, 255, 255)
                    
                    # Draw center
                    cv2.circle(display, tag.center, 8, color, -1)
                    
                    # Draw ID and distance
                    label = f"ID:{tag.tag_id} {tag.distance:.2f}m"
                    cv2.putText(display, label,
                               (tag.center[0] + 15, tag.center[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw bearing
                    bearing_deg = np.degrees(tag.bearing)
                    cv2.putText(display, f"{bearing_deg:.1f}°",
                               (tag.center[0] + 15, tag.center[1] + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw navigation state
                state_color = {
                    NavigationState.IDLE: (200, 200, 200),
                    NavigationState.DETECTING_TAGS: (255, 255, 0),
                    NavigationState.NAVIGATING: (0, 255, 0),
                    NavigationState.OBSTRUCTED: (0, 0, 255),
                    NavigationState.TARGET_REACHED: (255, 0, 255),
                    NavigationState.ERROR: (0, 0, 255)
                }.get(self.navigation_state, (200, 200, 200))
                
                cv2.putText(display, f"State: {self.navigation_state.value}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
                # Draw diagnostics
                if show_diagnostics:
                    diag = self.get_diagnostics()
                    y_offset = 60
                    for key, value in diag.items():
                        if key != 'navigation_state':
                            label = f"{key}: {value}"
                            if isinstance(value, float):
                                label = f"{key}: {value:.3f}"
                            cv2.putText(display, label,
                                       (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_offset += 20
                
                # Draw control commands
                cmd_y = h - 60
                cv2.putText(display, f"Accel: {self.current_command.acceleration:.3f} m/s²",
                           (10, cmd_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, f"Steer: {self.current_command.steering_rate:.3f} rad/s",
                           (10, cmd_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display
                cv2.imshow("Autonomous Navigation", display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    # Set target to first detected tag
                    if ground_tags:
                        self.set_target_tag(ground_tags[0].tag_id)
                        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
            cv2.destroyAllWindows()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Car Navigation with AprilTags')
    parser.add_argument('--target', type=int, default=None,
                       help='Target AprilTag ID to navigate to')
    parser.add_argument('--robot-width', type=float, default=0.5,
                       help='Robot width in meters')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create navigator
    navigator = AutonomousNavigator(
        robot_width=args.robot_width,
        target_tag_id=args.target
    )
    
    # Start system
    navigator.start()
    
    # Add some example known tags (optional - can be learned online)
    # navigator.add_known_tag(0, x=5.0, y=10.0)
    # navigator.add_known_tag(1, x=10.0, y=5.0)
    
    if args.no_display:
        # Run without visualization
        print("Running without visualization. Press Ctrl+C to stop.")
        try:
            while True:
                command = navigator.process_frame()
                
                if command:
                    diag = navigator.get_diagnostics()
                    print(f"[{diag['navigation_state']}] "
                          f"Pos: ({diag['estimated_position'][0]:.2f}, {diag['estimated_position'][1]:.2f}), "
                          f"Cmd: a={command.acceleration:.3f}, s={command.steering_rate:.3f}")
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    else:
        # Run with visualization
        navigator.run_visualization()


if __name__ == "__main__":
    main()
