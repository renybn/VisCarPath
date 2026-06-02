"""
Main Autonomous Navigation Pipeline
Integrates perception, state estimation, and control.

Modes:
  --visual    : live RGB + depth + obstacle overlay windows
  --headless  : no display; saves annotated debug images every LOG_INTERVAL frames
  (default)   : no display, no logging — bare control loop for deployment
"""
import os
# FIX: must be set before torch/ultralytics imports; prevents display-server
# connection attempts (xcb / Wayland) that block or crash on headless Pi.
os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import numpy as np
import time
import warnings
import depthai as dai
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from apriltag_detection import AprilTagDetector, AprilTagDetection
from ground_obstacle_detection import GroundAndObstaclePipeline
from kalman_filter import ExtendedKalmanFilter, TagMeasurementFusion, VehicleState
from mpc_controller import PathFollowingController, ControllerConfig
from vesc_bridge import VESCBridge

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
APRILTAG_INTERVAL = 3       # Run tag detection every N frames
GROUND_INTERVAL   = 2       # Run ground/obstacle perception every N frames
LOG_INTERVAL      = 20      # Save debug image every N frames (headless mode)
PIPELINE_MODE     = "optimized"  # "optimized" | "high_accuracy"
DEPTH_MEDIAN_K    = 3       # medianBlur kernel on raw depth


class NavigationState(Enum):
    IDLE           = "idle"
    DETECTING_TAGS = "detecting_tags"
    NAVIGATING     = "navigating"
    OBSTRUCTED     = "obstructed"
    TARGET_REACHED = "target_reached"


@dataclass
class NavigationCommand:
    acceleration:  float
    steering_rate: float
    target_velocity: float
    timestamp:     float


# ---------------------------------------------------------------------------
# OAK-D pipeline factory
# FIX: added confidence stream output so ObstacleDetector can mask
#      low-quality stereo pixels and reduce phantom obstacles.
# ---------------------------------------------------------------------------
def _build_oakd_pipeline(mode: str = PIPELINE_MODE) -> dai.Pipeline:
    pipeline  = dai.Pipeline()
    cam_rgb   = pipeline.create(dai.node.ColorCamera)
    mono_l    = pipeline.create(dai.node.MonoCamera)
    mono_r    = pipeline.create(dai.node.MonoCamera)
    stereo    = pipeline.create(dai.node.StereoDepth)

    cam_rgb.setPreviewSize(640, 400)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(30)
    cam_rgb.initialControl.setAutoFocusMode(
        dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)

    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_l.setFps(30)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_r.setFps(30)

    if mode == "high_accuracy":
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.initialConfig.setExtendedDisparity(True)
        stereo.initialConfig.setSubpixel(True)
        stereo.initialConfig.setConfidenceThreshold(200)
        stereo.setLeftRightCheck(True)
    else:  # optimized
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.initialConfig.setExtendedDisparity(False)
        stereo.initialConfig.setSubpixel(False)
        stereo.initialConfig.setConfidenceThreshold(220)
        stereo.setLeftRightCheck(True)

    stereo.setOutputSize(640, 400)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_rgb   = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    # FIX: confidence stream was missing — required for confidence-map filtering
    xout_conf = pipeline.create(dai.node.XLinkOut)
    xout_conf.setStreamName("confidence")
    stereo.confidenceMap.link(xout_conf.input)

    return pipeline


# ---------------------------------------------------------------------------
# AutonomousNavigator
# ---------------------------------------------------------------------------
class AutonomousNavigator:
    def __init__(self, robot_width: float = 0.5,
                 target_tag_id: Optional[int] = None,
                 fastsam_model: str = "FastSAM-s.pt"):
        self.tag_detector    = AprilTagDetector(tag_family="tag36h11",
                                                quad_decimate=2.0)
        self.ground_pipeline = GroundAndObstaclePipeline(robot_width=robot_width,
                                                          fastsam_model=fastsam_model)

        initial_state = VehicleState(x=0, y=0, theta=0, v=0, omega=0)
        self.ekf       = ExtendedKalmanFilter(initial_state=initial_state,
                                               process_noise=0.1,
                                               measurement_noise=0.5)
        self.tag_fusion = TagMeasurementFusion(self.ekf)

        ctrl_config    = ControllerConfig(dt=0.1, max_velocity=1.5,
                                           max_acceleration=0.8)
        self.controller = PathFollowingController(ctrl_config)

        self.navigation_state = NavigationState.IDLE
        self.current_command  = NavigationCommand(0, 0, 0, time.time())
        self.target_tag_id    = target_tag_id
        self.tag_map: Dict[int, tuple] = {}
        self.vesc: Optional[VESCBridge] = None

        # Camera / device handles
        self.device:   Optional[dai.Device] = None
        self.q_rgb     = None
        self.q_depth   = None
        self.q_conf    = None          # FIX: confidence queue
        self.K         = np.array([[552.6, 0, 311.4],
                                    [0, 552.6, 200.0],
                                    [0, 0, 1]], dtype=np.float32)

        # Cached frame state
        self.last_rgb:   Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.last_conf:  Optional[np.ndarray] = None   # FIX: cached confidence
        self.last_tags:  List[AprilTagDetection] = []
        self.last_obstacles: list = []
        self.last_ground = None

        # FPS
        self.fps             = 0
        self._frame_count    = 0
        self._fps_timer      = time.time()

        # Per-type skip counters
        self._tag_frame      = 0
        self._ground_frame   = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_target_tag(self, tag_id: int):
        self.target_tag_id = tag_id

    def add_known_tag(self, tag_id: int, x: float, y: float, theta: float = 0):
        self.tag_map[tag_id] = (x, y, theta)
        self.tag_fusion.add_tag_to_map(tag_id, x, y, theta)

    # ------------------------------------------------------------------
    # Hardware lifecycle
    # ------------------------------------------------------------------
    def start(self):
        pipeline    = _build_oakd_pipeline(PIPELINE_MODE)
        self.device = dai.Device(pipeline, usb2Mode=True)

        # Non-blocking queues; maxSize=2 to absorb one frame of jitter
        self.q_rgb   = self.device.getOutputQueue("rgb",        maxSize=2, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth",      maxSize=2, blocking=False)
        # FIX: open confidence queue
        self.q_conf  = self.device.getOutputQueue("confidence", maxSize=2, blocking=False)

        calib      = self.device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)
        self.K     = np.array(intrinsics, dtype=np.float32)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        self.tag_detector.set_camera_intrinsics(fx, fy, cx, cy)
        self.ground_pipeline.set_camera_intrinsics(fx, fy, cx, cy)

        self.navigation_state = NavigationState.DETECTING_TAGS
        print(f"[NAV] OAK-D ready | fx={fx:.1f} fy={fy:.1f} | mode={PIPELINE_MODE}")
        self.vesc = VESCBridge(port='/dev/ttyACM0')

    def stop(self):
        if self.vesc:
            self.vesc.close()
        if self.device is not None:
            self.device.close()
            self.device = None

    # ------------------------------------------------------------------
    # Frame capture (non-blocking)
    # ------------------------------------------------------------------
    def _capture(self):
        """Returns (rgb, depth, conf) or (None, None, None) if no frame ready."""
        if self.q_rgb is None:
            return None, None, None
        rgb_pkt   = self.q_rgb.tryGet()
        depth_pkt = self.q_depth.tryGet()
        if rgb_pkt is None or depth_pkt is None:
            return None, None, None
        rgb   = rgb_pkt.getCvFrame()
        depth = cv2.medianBlur(depth_pkt.getFrame(), DEPTH_MEDIAN_K)
        # FIX: read confidence non-blocking; returns None if not ready (handled below)
        conf_pkt = self.q_conf.tryGet() if self.q_conf else None
        conf     = conf_pkt.getFrame() if conf_pkt is not None else self.last_conf
        return rgb, depth, conf

    # ------------------------------------------------------------------
    # Core per-frame logic
    # ------------------------------------------------------------------
    def process_frame(self) -> Optional[NavigationCommand]:
        rgb_frame, depth_frame, conf_frame = self._capture()
        if rgb_frame is None:
            return None

        self.last_rgb   = rgb_frame
        self.last_depth = depth_frame
        self.last_conf  = conf_frame   # FIX: persist latest confidence
        h, w = rgb_frame.shape[:2]

        # -- AprilTag detection (gated) --
        self._tag_frame += 1
        if self._tag_frame % APRILTAG_INTERVAL == 0:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            self.last_tags = self.tag_detector.detect_tags(gray, depth_frame)

        tag_detections = self.last_tags

        # -- Target selection --
        target_tag = None
        if self.target_tag_id is not None:
            target_tag = next(
                (t for t in tag_detections if t.tag_id == self.target_tag_id), None)
        elif tag_detections:
            target_tag = min(tag_detections, key=lambda t: t.distance)
            self.target_tag_id = target_tag.tag_id

        # -- Static landmark mapping --
        static_tags = [t for t in tag_detections if t.tag_id != self.target_tag_id]
        for tag in static_tags:
            if tag.tag_id not in self.tag_map:
                state  = self.ekf.get_state()
                cam_x, cam_z = tag.pose[0, 3], tag.pose[2, 3]
                wx = state.x + cam_z * np.cos(state.theta) - cam_x * np.sin(state.theta)
                wy = state.y + cam_z * np.sin(state.theta) + cam_x * np.cos(state.theta)
                self.add_known_tag(tag.tag_id, wx, wy)

        # -- Ground & obstacle perception with FastSAM fusion (gated) --
        self._ground_frame += 1
        if self._ground_frame % GROUND_INTERVAL == 0:
            # Convert RGB→BGR once here; ground_pipeline passes it to FastSAM
            rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            frame_data = self.ground_pipeline.process_frame(
                depth_frame, rgb_bgr, tag_detections, (h, w),
                confidence_map=conf_frame)
            self.last_ground    = frame_data['ground_plane']
            self.last_obstacles = frame_data['obstacles']

        ground_plane = self.last_ground
        obstacles    = self.last_obstacles

        # -- EKF update + predict --
        if static_tags:
            self.tag_fusion.update_ekf_with_tags(static_tags)

        accel_prev = self.current_command.acceleration
        steer_prev = self.current_command.steering_rate
        self.ekf.predict(dt=0.1, control_input=(accel_prev, steer_prev))
        current_state = self.ekf.get_state()

        # -- Path planning --
        path = None
        if target_tag and ground_plane:
            path = self.ground_pipeline.path_planner.plan_path_to_tag(
                target_tag, obstacles, ground_plane,
                self.ground_pipeline.camera_intrinsics, (h, w))

        # -- State machine --
        old_state = self.navigation_state
        if target_tag is None:
            self.navigation_state = NavigationState.DETECTING_TAGS
        elif target_tag.distance < 0.5:
            self.navigation_state = NavigationState.TARGET_REACHED
        elif obstacles and min(o.distance for o in obstacles) < 1.0:
            self.navigation_state = NavigationState.OBSTRUCTED
        else:
            self.navigation_state = NavigationState.NAVIGATING

        if old_state != self.navigation_state:
            print(f"[NAV] {old_state.value} -> {self.navigation_state.value}")

        # -- Control --
        if self.navigation_state == NavigationState.NAVIGATING:
            state_arr = np.array([current_state.x, current_state.y,
                                    current_state.theta, current_state.v, 0.0])
            self.controller.update_state(state_arr)
            if path:
                try:
                    accel, steer = self.controller.compute_control(path, obstacles)
                except Exception as e:
                    print(f"[NAV] Control error: {e}")
                    accel, steer = 0.0, 0.0
                self.current_command = NavigationCommand(
                    accel, steer, current_state.v + accel * 0.1, time.time())
            else:
                self.current_command = NavigationCommand(0, 0, 0, time.time())
        else:
            self.current_command = NavigationCommand(0, 0, 0, time.time())

        self._update_fps()
        return self.current_command

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        if now - self._fps_timer >= 1.0:
            self.fps          = self._frame_count
            self._frame_count = 0
            self._fps_timer   = now

    def get_diagnostics(self) -> Dict[str, Any]:
        state = self.ekf.get_state()
        return {
            'state':   self.navigation_state.value,
            'pos':     (state.x, state.y),
            'heading': np.degrees(state.theta),
            'vel':     state.v,
            'target':  self.target_tag_id,
            'fps':     self.fps,
            'accel':   self.current_command.acceleration,
            'steer':   self.current_command.steering_rate,
        }

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _draw_rgb_overlay(self, rgb_bgr: np.ndarray,
                           command: NavigationCommand) -> np.ndarray:
        h, w = rgb_bgr.shape[:2]

        STATE_COLORS = {
            NavigationState.IDLE:           (200, 200, 200),
            NavigationState.DETECTING_TAGS: (255, 255,   0),
            NavigationState.NAVIGATING:     (  0, 255,   0),
            NavigationState.OBSTRUCTED:     (  0,   0, 255),
            NavigationState.TARGET_REACHED: (255,   0, 255),
        }
        sc = STATE_COLORS.get(self.navigation_state, (200, 200, 200))

        # Tags
        for tag in self.last_tags:
            if np.isnan(tag.distance):
                continue
            cx_t, cy_t = tag.center
            if not (0 < cx_t < w and 0 < cy_t < h):
                continue
            color = (0, 255, 0) if tag.tag_id == self.target_tag_id else (0, 255, 255)
            cv2.polylines(rgb_bgr, [tag.corners.astype(int)], True, color, 2)
            cv2.circle(rgb_bgr, (cx_t, cy_t), 6, color, -1)
            cv2.putText(rgb_bgr, f"ID:{tag.tag_id} {float(tag.distance):.2f}m",
                        (cx_t + 10, cy_t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Obstacles — filled polygon + outline, severity-tinted red→yellow
        overlay = rgb_bgr.copy()
        for obs in self.last_obstacles:
            if obs.contour is None:
                continue
            t     = float(np.clip(obs.severity, 0.0, 1.0))
            color = (0, int(200 * (1 - t)), int(255 * t + 200 * (1 - t)))
            cnt_scaled = obs.contour.astype(np.float32)
            cnt_scaled[:, :, 0] *= w / 320.0
            cnt_scaled[:, :, 1] *= h / 240.0
            cnt_scaled = cnt_scaled.astype(np.int32)
            cv2.drawContours(overlay, [cnt_scaled], -1, color, thickness=cv2.FILLED)
            cv2.drawContours(rgb_bgr, [cnt_scaled], -1, (255, 255, 255), 1)
            u, v = obs.center
            u_s  = int(u * w / 320.0)
            v_s  = int(v * h / 240.0)
            cv2.putText(rgb_bgr, f"{obs.distance:.2f}m",
                        (u_s + 4, v_s - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.35, rgb_bgr, 0.65, 0, rgb_bgr)

        # HUD
        cv2.putText(rgb_bgr, f"State: {self.navigation_state.value}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)
        diag  = self.get_diagnostics()
        y_off = 50
        for k in ('fps', 'vel', 'heading'):
            v_str = (f"{diag[k]:.1f}" if isinstance(diag[k], float) else str(diag[k]))
            cv2.putText(rgb_bgr, f"{k}: {v_str}",
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_off += 18
        cv2.putText(rgb_bgr,
                    f"accel:{command.acceleration:.2f} steer:{command.steering_rate:.2f}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return rgb_bgr

    def _make_depth_viz(self, depth: np.ndarray) -> np.ndarray:
        clipped = np.clip(depth.astype(np.float32), 100, 3000)
        norm    = cv2.normalize(clipped, None, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # ------------------------------------------------------------------
    # Run modes
    # ------------------------------------------------------------------
    def run_visual(self):
        print("[NAV] Visual mode. Press 'q' to quit, 't' to lock first tag as target.")
        try:
            while True:
                command = self.process_frame()

                if command and self.vesc:
                    self.vesc.send_command(command.acceleration, command.steering_rate)

                if self.last_rgb is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                if command is None:
                    command = self.current_command

                display = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
                self._draw_rgb_overlay(display, command)

                depth_viz = self._make_depth_viz(self.last_depth)
                dh, dw    = depth_viz.shape[:2]
                rh, rw    = display.shape[:2]
                if dh != rh:
                    depth_viz = cv2.resize(depth_viz, (int(dw * rh / dh), rh))

                combined = np.hstack([display, depth_viz])
                cv2.imshow("Navigation | RGB + Depth", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t') and self.last_tags:
                    self.set_target_tag(self.last_tags[0].tag_id)
                    print(f"[NAV] Target locked -> ID {self.last_tags[0].tag_id}")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def run_headless(self, log_dir: str = "nav_logs"):
        os.makedirs(log_dir, exist_ok=True)
        print(f"[NAV] Headless mode. Logs -> {log_dir}/  Press Ctrl+C to stop.")
        frame_idx = 0
        try:
            while True:
                command = self.process_frame()

                if command and self.vesc:
                    self.vesc.send_command(command.acceleration, command.steering_rate)

                if command is None:
                    # FIX: yield CPU when no camera frame is ready.
                    # Busy-spinning starves DepthAI's USB thread and triggers
                    # ARM thermal throttling, collapsing headless FPS to <1.
                    time.sleep(0.005)
                    continue

                frame_idx += 1

                if frame_idx % LOG_INTERVAL == 0 and self.last_rgb is not None:
                    display   = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
                    self._draw_rgb_overlay(display, command)
                    depth_viz = self._make_depth_viz(self.last_depth)
                    dh, dw    = depth_viz.shape[:2]
                    rh, _     = display.shape[:2]
                    if dh != rh:
                        depth_viz = cv2.resize(depth_viz, (int(dw * rh / dh), rh))
                    combined  = np.hstack([display, depth_viz])
                    path_out  = os.path.join(log_dir, f"frame_{frame_idx:05d}.jpg")
                    cv2.imwrite(path_out, combined)

                # FIX: gate diagnostic print to every LOG_INTERVAL frames.
                # Printing every frame (~30 writes/sec) was a significant
                # source of headless latency via blocking stdout syscalls.
                if frame_idx % LOG_INTERVAL == 0:
                    diag = self.get_diagnostics()
                    print(f"[{diag['state']:16s}] FPS:{diag['fps']:2d} "
                          f"pos=({diag['pos'][0]:.2f},{diag['pos'][1]:.2f}) "
                          f"hdg={diag['heading']:.1f}° "
                          f"a={command.acceleration:.3f} s={command.steering_rate:.3f}")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def run_bare(self):
        print("[NAV] Bare mode. Press Ctrl+C to stop.")
        try:
            while True:
                command = self.process_frame()
                if command is not None:
                    if self.vesc:
                        self.vesc.send_command(command.acceleration, command.steering_rate)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous RC Car Navigation")
    parser.add_argument("--target",       type=int,   default=None,
                        help="Target AprilTag ID")
    parser.add_argument("--robot-width",  type=float, default=0.5,
                        help="Robot width in metres")
    parser.add_argument("--visual",       action="store_true",
                        help="Show live RGB + depth windows")
    parser.add_argument("--headless",     action="store_true",
                        help="Save debug images to disk, no display")
    parser.add_argument("--log-dir",      type=str,   default="nav_logs",
                        help="Directory for headless debug images")
    parser.add_argument("--fastsam-model", type=str, default="FastSAM-s.pt",
                        help="Path to FastSAM weights file (default: FastSAM-s.pt)")
    args = parser.parse_args()

    navigator = AutonomousNavigator(
        robot_width=args.robot_width,
        target_tag_id=args.target,
        fastsam_model=args.fastsam_model)
    navigator.start()

    if args.visual:
        navigator.run_visual()
    elif args.headless:
        navigator.run_headless(log_dir=args.log_dir)
    else:
        navigator.run_bare()


if __name__ == "__main__":
    main()