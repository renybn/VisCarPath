"""
Main Autonomous Navigation Pipeline
Integrates perception, state estimation, and control.
"""
import cv2
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from apriltag_detection import OakDAprilTagPipeline
from ground_obstacle_detection import GroundAndObstaclePipeline
from kalman_filter import ExtendedKalmanFilter, TagMeasurementFusion, VehicleState
from mpc_controller import PathFollowingController, ControllerConfig

class NavigationState(Enum):
    IDLE = "idle"
    DETECTING_TAGS = "detecting_tags"
    NAVIGATING = "navigating"
    OBSTRUCTED = "obstructed"
    TARGET_REACHED = "target_reached"

@dataclass
class NavigationCommand:
    acceleration: float
    steering_rate: float
    target_velocity: float
    timestamp: float

class AutonomousNavigator:
    def __init__(self, robot_width: float = 0.5, target_tag_id: Optional[int] = None):
        self.oak_pipeline = OakDAprilTagPipeline()
        self.ground_pipeline = GroundAndObstaclePipeline(robot_width=robot_width)
        
        initial_state = VehicleState(x=0, y=0, theta=0, v=0, omega=0)
        self.ekf = ExtendedKalmanFilter(initial_state=initial_state, process_noise=0.1, measurement_noise=0.5)
        self.tag_fusion = TagMeasurementFusion(self.ekf)
        
        ctrl_config = ControllerConfig(dt=0.1, max_velocity=1.5, max_acceleration=0.8)
        self.controller = PathFollowingController(ctrl_config)
        
        self.navigation_state = NavigationState.IDLE
        self.current_command = NavigationCommand(0, 0, 0, time.time())
        self.target_tag_id = target_tag_id
        self.tag_map: Dict[int, tuple] = {}
        
        self.last_rgb = None
        self.last_depth = None
        self.last_tags = []
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        from imu_integration import ThreadedIMU
        self.imu = ThreadedIMU(use_rotation_vector=True)


    def set_target_tag(self, tag_id: int):
        self.target_tag_id = tag_id

    def add_known_tag(self, tag_id: int, x: float, y: float, theta: float = 0):
        self.tag_map[tag_id] = (x, y, theta)
        self.tag_fusion.add_tag_to_map(tag_id, x, y, theta)

    def start(self):
        self.oak_pipeline.start()
        if self.oak_pipeline.device is not None:
            self.imu.start(self.oak_pipeline.device)
        ratio = 480 / 1080
        fx = self.oak_pipeline.april_detector.fx * ratio
        fy = self.oak_pipeline.april_detector.fy * ratio
        cx = self.oak_pipeline.april_detector.cx * ratio
        cy = self.oak_pipeline.april_detector.cy * ratio
        self.ground_pipeline.set_camera_intrinsics(fx, fy, cx, cy)
        self.navigation_state = NavigationState.DETECTING_TAGS

    def stop(self):
        if self.oak_pipeline.device:
            self.oak_pipeline.stop()
        self.navigation_state = NavigationState.IDLE
        self.imu.stop()


    def process_frame(self) -> Optional[NavigationCommand]:
        rgb_frame, depth_frame, _ = self.oak_pipeline.get_frame_data()
        if rgb_frame is None or depth_frame is None:
            return None
            
        h, w = rgb_frame.shape[:2]
        tag_detections = self.oak_pipeline.detect_tags_in_frame(rgb_frame, depth_frame)
        self.last_rgb, self.last_depth, self.last_tags = rgb_frame, depth_frame, tag_detections
        
        # 1. TARGET SELECTION
        target_tag = None
        if self.target_tag_id is not None:
            target_tag = next((t for t in tag_detections if t.tag_id == self.target_tag_id), None)
        elif tag_detections:
            target_tag = min(tag_detections, key=lambda t: t.distance)
            self.target_tag_id = target_tag.tag_id
            
        # 2. STATIC LANDMARK FILTERING
        # Exclude target tag from mapping and EKF to prevent moving target from corrupting localization
        static_tags = [t for t in tag_detections if t.tag_id != self.target_tag_id]
        
        for tag in static_tags:
            if tag.tag_id not in self.tag_map:
                state = self.ekf.get_state()
                cam_x, cam_z = tag.pose[0, 3], tag.pose[2, 3]
                # 2D rotation matching TagMeasurementFusion logic
                tag_world_x = state.x + cam_z * np.cos(state.theta) - cam_x * np.sin(state.theta)
                tag_world_y = state.y + cam_z * np.sin(state.theta) + cam_x * np.cos(state.theta)
                self.add_known_tag(tag.tag_id, tag_world_x, tag_world_y)
                
        # 3. PERCEPTION (Ground & Obstacles)
        frame_data = self.ground_pipeline.process_frame(depth_frame, tag_detections, (h, w))
        ground_plane = frame_data['ground_plane']
        obstacles = frame_data['obstacles']
        
        # 4. STATE ESTIMATION (EKF)
        if static_tags:
            self.tag_fusion.update_ekf_with_tags(static_tags)
            
        imu_state = self.imu.get_state()
        imu_yaw_rate = imu_state[1]  # rad/s from gyroscope/fusion
        accel_cmd = self.current_command.acceleration

        # Predict with measured yaw rate, not commanded steering
        self.ekf.predict(
            dt=0.1,
            control_input=(accel_cmd, imu_yaw_rate)
        )
        
        current_state = self.ekf.get_state()
        
        # 5. REACTIVE PATH PLANNING (Camera-frame)
        path = None
        if target_tag and ground_plane:
            path = self.ground_pipeline.path_planner.plan_path_to_tag(
                target_tag, obstacles, ground_plane,
                self.ground_pipeline.camera_intrinsics, (h, w)
            )
            
        # 6. STATE MACHINE
        if target_tag is None:
            self.navigation_state = NavigationState.DETECTING_TAGS
        elif target_tag.distance < 0.5:
            self.navigation_state = NavigationState.TARGET_REACHED
        elif obstacles:
            closest_obs = min(obstacles, key=lambda o: o.distance)
            self.navigation_state = NavigationState.OBSTRUCTED if closest_obs.distance < 1.0 else NavigationState.NAVIGATING
        else:
            self.navigation_state = NavigationState.NAVIGATING
            
        # 7. CONTROL COMPUTATION
        if self.navigation_state == NavigationState.NAVIGATING:
            state_array = np.array([current_state.x, current_state.y, current_state.theta, current_state.v, 0.0])
            self.controller.update_state(state_array)
            
            if path:
                try:
                    accel, steer = self.controller.compute_control(path, obstacles)
                except Exception:
                    accel, steer = 0.0, 0.0
                self.current_command = NavigationCommand(accel, steer, current_state.v + accel * 0.1, time.time())
            else:
                self.current_command = NavigationCommand(0, 0, 0, time.time())
        else:
            self.current_command = NavigationCommand(0, 0, 0, time.time())
            
        self._update_fps()
        return self.current_command

    def _update_fps(self):
        self.frame_count += 1
        if time.time() - self.last_frame_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_frame_time = time.time()

    def get_diagnostics(self) -> Dict[str, Any]:
        state = self.ekf.get_state()
        return {
            'state': self.navigation_state.value,
            'pos': (state.x, state.y),
            'heading': np.degrees(state.theta),
            'vel': state.v,
            'target_id': self.target_tag_id,
            'fps': self.fps,
            'cmd_a': self.current_command.acceleration,
            'cmd_s': self.current_command.steering_rate
        }

    # ... [run_visualization and main remain unchanged] ...
    def run_visualization(self, show_diagnostics: bool = True):
        """Run navigation with stable visualization"""
        print("Starting visualization mode. Press 'q' to quit.")
        
        try:
            while True:
                # 1. Process heavy math
                command = self.process_frame()
                
                # 2. Skip drawing if no frame ready yet
                if self.last_rgb is None:
                    # CRITICAL: Keep OpenCV window alive even when skipping
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # 3. Prepare display
                display = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
                h, w = display.shape[:2]
                ground_tags = self.last_tags
                
                # 4. Draw detections (with bounds checking & safe formatting)
                for tag in ground_tags:
                    # Skip invalid/nan detections
                    if np.isnan(tag.distance) or not (0 < tag.center[0] < w and 0 < tag.center[1] < h):
                        continue
                        
                    color = (0, 255, 0) if tag.tag_id == self.target_tag_id else (0, 255, 255)
                    cv2.circle(display, tag.center, 8, color, -1)
                    
                    # CRITICAL: Wrap numpy values in float() for f-string formatting
                    dist = float(tag.distance)
                    bearing_deg = float(np.degrees(np.squeeze(tag.bearing)))
                    
                    label = f"ID:{tag.tag_id} {dist:.2f}m"
                    cv2.putText(display, label, (tag.center[0] + 15, tag.center[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    cv2.putText(display, f"{bearing_deg:.1f}°",
                               (tag.center[0] + 15, tag.center[1] + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 5. Draw navigation state
                state_color = {
                    NavigationState.IDLE: (200, 200, 200),
                    NavigationState.DETECTING_TAGS: (255, 255, 0),
                    NavigationState.NAVIGATING: (0, 255, 0),
                    NavigationState.OBSTRUCTED: (0, 0, 255),
                    NavigationState.TARGET_REACHED: (255, 0, 255),
                }.get(self.navigation_state, (200, 200, 200))
                
                cv2.putText(display, f"State: {self.navigation_state.value}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
                # 6. Draw diagnostics & commands
                if show_diagnostics:
                    diag = self.get_diagnostics()
                    y_offset = 60
                    for key, value in diag.items():
                        if key != 'navigation_state':
                            val_str = f"{value:.3f}" if isinstance(value, float) else str(value)
                            cv2.putText(display, f"{key}: {val_str}",
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_offset += 20
                
                cmd_y = h - 40
                cv2.putText(display, f"Accel: {command.acceleration:.2f} | Steer: {command.steering_rate:.2f}",
                           (10, cmd_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 7. Display & handle input
                cv2.imshow("Autonomous Navigation", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t') and ground_tags:
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
