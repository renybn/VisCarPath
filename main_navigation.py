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
        print("[NAVIGATOR] Initializing AutonomousNavigator...")
        self.oak_pipeline = OakDAprilTagPipeline()
        print("[NAVIGATOR]   - OAK-D AprilTag pipeline created")
        self.ground_pipeline = GroundAndObstaclePipeline(robot_width=robot_width)
        print(f"[NAVIGATOR]   - Ground/obstacle pipeline created (robot_width={robot_width}m)")
        
        initial_state = VehicleState(x=0, y=0, theta=0, v=0, omega=0)
        self.ekf = ExtendedKalmanFilter(initial_state=initial_state, process_noise=0.1, measurement_noise=0.5)
        print("[NAVIGATOR]   - Extended Kalman Filter initialized at origin (0,0,0)")
        self.tag_fusion = TagMeasurementFusion(self.ekf)
        print("[NAVIGATOR]   - Tag measurement fusion module ready")
        
        ctrl_config = ControllerConfig(dt=0.1, max_velocity=1.5, max_acceleration=0.8)
        self.controller = PathFollowingController(ctrl_config)
        print("[NAVIGATOR]   - Path following controller configured")
        
        self.navigation_state = NavigationState.IDLE
        self.current_command = NavigationCommand(0, 0, 0, time.time())
        self.target_tag_id = target_tag_id
        self.tag_map: Dict[int, tuple] = {}
        print(f"[NAVIGATOR]   - Navigation state: IDLE, Target tag: {target_tag_id}")
        
        self.last_rgb = None
        self.last_depth = None
        self.last_tags = []
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        


    def set_target_tag(self, tag_id: int):
        self.target_tag_id = tag_id

    def add_known_tag(self, tag_id: int, x: float, y: float, theta: float = 0):
        self.tag_map[tag_id] = (x, y, theta)
        self.tag_fusion.add_tag_to_map(tag_id, x, y, theta)

    def start(self):
        print("[NAVIGATOR] Starting navigation system...")
        self.oak_pipeline.start()
        ratio = 480 / 1080
        fx = self.oak_pipeline.april_detector.fx
        fy = self.oak_pipeline.april_detector.fy
        cx = self.oak_pipeline.april_detector.cx
        cy = self.oak_pipeline.april_detector.cy
        self.ground_pipeline.set_camera_intrinsics(fx, fy, cx, cy)
        print(f"[NAVIGATOR]   - Camera intrinsics set for ground pipeline: fx={fx:.1f}, fy={fy:.1f}")
        self.navigation_state = NavigationState.DETECTING_TAGS
        print(f"[NAVIGATOR]   - Navigation state changed to: DETECTING_TAGS")
        print("[NAVIGATOR] System started successfully!\n")

    def stop(self):
        print("[NAVIGATOR] Stopping navigation system...")
        if self.oak_pipeline.device:
            self.oak_pipeline.stop()
            print("[NAVIGATOR]   - OAK-D pipeline stopped")
        self.navigation_state = NavigationState.IDLE


    def process_frame(self) -> Optional[NavigationCommand]:
        """Process a single frame and return navigation command"""
        # Get sensor data
        rgb_frame, depth_frame, _ = self.oak_pipeline.get_frame_data()
        if rgb_frame is None or depth_frame is None:
            print("[PROCESS] Skipping frame - no RGB/Depth data from OAK-D")
            return None
        
        h, w = rgb_frame.shape[:2]
        print(f"[PROCESS] Processing frame {h}x{w}...")
        
        # Detect AprilTags
        tag_detections = self.oak_pipeline.detect_tags_in_frame(rgb_frame, depth_frame)
        self.last_rgb, self.last_depth, self.last_tags = rgb_frame, depth_frame, tag_detections
        print(f"[PROCESS]   - Detected {len(tag_detections)} AprilTag(s)")
        
        # 1. TARGET SELECTION
        target_tag = None
        if self.target_tag_id is not None:
            target_tag = next((t for t in tag_detections if t.tag_id == self.target_tag_id), None)
            if target_tag:
                print(f"[PROCESS]   - Target tag {self.target_tag_id} found at {target_tag.distance:.2f}m")
            else:
                print(f"[PROCESS]   - Target tag {self.target_tag_id} NOT detected")
        elif tag_detections:
            target_tag = min(tag_detections, key=lambda t: t.distance)
            self.target_tag_id = target_tag.tag_id
            print(f"[PROCESS]   - No target set, selecting closest tag: ID={target_tag.tag_id} at {target_tag.distance:.2f}m")
            
        # 2. STATIC LANDMARK FILTERING
        # Exclude target tag from mapping and EKF to prevent moving target from corrupting localization
        static_tags = [t for t in tag_detections if t.tag_id != self.target_tag_id]
        if static_tags:
            print(f"[PROCESS]   - Found {len(static_tags)} static tag(s) for localization")
        
        for tag in static_tags:
            if tag.tag_id not in self.tag_map:
                state = self.ekf.get_state()
                cam_x, cam_z = tag.pose[0, 3], tag.pose[2, 3]
                # 2D rotation matching TagMeasurementFusion logic
                tag_world_x = state.x + cam_z * np.cos(state.theta) - cam_x * np.sin(state.theta)
                tag_world_y = state.y + cam_z * np.sin(state.theta) + cam_x * np.cos(state.theta)
                self.add_known_tag(tag.tag_id, tag_world_x, tag_world_y)
                print(f"[PROCESS]     - Mapped new static tag {tag.tag_id} at world position ({tag_world_x:.2f}, {tag_world_y:.2f})")
            
        # 3. PERCEPTION (Ground & Obstacles)
        frame_data = self.ground_pipeline.process_frame(depth_frame, tag_detections, (h, w))
        ground_plane = frame_data['ground_plane']
        obstacles = frame_data['obstacles']
        
        if ground_plane:
            print(f"[PROCESS]   - Ground plane detected: confidence={ground_plane.confidence:.2f}, {ground_plane.inliers} inliers")
        else:
            print("[PROCESS]   - WARNING: No ground plane detected")
        
        if obstacles:
            print(f"[PROCESS]   - Detected {len(obstacles)} obstacle(s), closest at {min(o.distance for o in obstacles):.2f}m")
        else:
            print("[PROCESS]   - No obstacles detected")
        
        # 4. STATE ESTIMATION (EKF)
        if static_tags:
            fusion_success = self.tag_fusion.update_ekf_with_tags(static_tags)
            if fusion_success:
                print("[PROCESS]   - EKF updated with static tag measurements")
            else:
                print("[PROCESS]   - EKF update skipped - no valid tag measurements")
            
        accel_cmd = self.current_command.acceleration
        steer_cmd = self.current_command.steering_rate

        # Predict using the commanded steering rate (bicycle model)
        self.ekf.predict(
            dt=0.1,
            control_input=(accel_cmd, steer_cmd)
        )
        
        current_state = self.ekf.get_state()
        print(f"[PROCESS]   - EKF prediction complete: pos=({current_state.x:.2f}, {current_state.y:.2f}), heading={np.degrees(current_state.theta):.1f}°, vel={current_state.v:.2f}m/s")
        
        # 5. REACTIVE PATH PLANNING (Camera-frame)
        path = None
        if target_tag and ground_plane:
            path = self.ground_pipeline.path_planner.plan_path_to_tag(
                target_tag, obstacles, ground_plane,
                self.ground_pipeline.camera_intrinsics, (h, w)
            )
            if path:
                print(f"[PROCESS]   - Path planned: {len(path)} segment(s), cost={path[0].cost:.2f}")
            else:
                print("[PROCESS]   - Path planning failed - too close to target or invalid geometry")
        elif not target_tag:
            print("[PROCESS]   - Path planning skipped - no target tag")
        elif not ground_plane:
            print("[PROCESS]   - Path planning skipped - no ground plane")
            
        # 6. STATE MACHINE
        old_state = self.navigation_state
        if target_tag is None:
            self.navigation_state = NavigationState.DETECTING_TAGS
        elif target_tag.distance < 0.5:
            self.navigation_state = NavigationState.TARGET_REACHED
        elif obstacles:
            closest_obs = min(obstacles, key=lambda o: o.distance)
            self.navigation_state = NavigationState.OBSTRUCTED if closest_obs.distance < 1.0 else NavigationState.NAVIGATING
        else:
            self.navigation_state = NavigationState.NAVIGATING
        
        if old_state != self.navigation_state:
            print(f"[PROCESS]   - State transition: {old_state.value} -> {self.navigation_state.value}")
            
        # 7. CONTROL COMPUTATION
        if self.navigation_state == NavigationState.NAVIGATING:
            state_array = np.array([current_state.x, current_state.y, current_state.theta, current_state.v, 0.0])
            self.controller.update_state(state_array)
            
            if path:
                try:
                    accel, steer = self.controller.compute_control(path, obstacles)
                    print(f"[PROCESS]   - Control computed: accel={accel:.3f}, steer={steer:.3f}")
                except Exception as e:
                    print(f"[PROCESS]   - Control computation FAILED: {e}")
                    accel, steer = 0.0, 0.0
                self.current_command = NavigationCommand(accel, steer, current_state.v + accel * 0.1, time.time())
            else:
                print("[PROCESS]   - No path available, commanding stop")
                self.current_command = NavigationCommand(0, 0, 0, time.time())
        else:
            print(f"[PROCESS]   - Not in NAVIGATING state ({self.navigation_state.value}), commanding stop")
            self.current_command = NavigationCommand(0, 0, 0, time.time())
            
        self._update_fps()
        print(f"[PROCESS] Frame processing complete. FPS: {self.fps}\n")
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
                    print(f"[{diag['state']}] "
                        f"Pos: ({diag['pos'][0]:.2f}, {diag['pos'][1]:.2f}), "
                        f"Cmd: a={command.acceleration:.3f}, s={command.steering_rate:.3f}")
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    else:
        # Run with visualization
        navigator.run_visualization()


if __name__ == "__main__":
    main()
