"""
Kalman Filter for Vehicle State Estimation
Estimates position, velocity, and orientation for autonomous navigation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class MotionModel(Enum):
    """Vehicle motion model types"""
    CONSTANT_VELOCITY = "cv"
    BICYCLE = "bicycle"


@dataclass
class VehicleState:
    """Complete vehicle state estimate"""
    x: float  # X position (meters, right from start)
    y: float  # Y position (meters, forward from start)
    theta: float  # Heading angle (radians, 0 = forward)
    v: float  # Linear velocity (m/s)
    omega: float  # Angular velocity (rad/s)
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array"""
        return np.array([self.x, self.y, self.theta, self.v, self.omega])
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Create state from numpy array"""
        return cls(
            x=arr[0], y=arr[1], theta=arr[2],
            v=arr[3], omega=arr[4]
        )


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for vehicle state estimation
    Uses bicycle model for prediction and AprilTag measurements for correction
    """
    
    def __init__(self, initial_state: Optional[VehicleState] = None,
                 process_noise: float = 0.1,
                 measurement_noise: float = 0.5):
        """
        Initialize EKF
        
        Args:
            initial_state: Initial vehicle state (defaults to origin)
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
        """
        # State vector: [x, y, theta, v, omega]
        if initial_state is None:
            self.x = np.zeros(5)
        else:
            self.x = initial_state.to_array()
        
        # State covariance matrix
        self.P = np.eye(5) * 1.0
        
        # Process noise covariance
        self.Q = np.diag([
            process_noise**2,      # x noise
            process_noise**2,      # y noise
            (process_noise * 0.5)**2,  # theta noise
            (process_noise * 2)**2,    # v noise
            (process_noise * 2)**2     # omega noise
        ])
        
        # Measurement noise covariance
        self.R = np.diag([
            measurement_noise**2,    # x measurement noise
            measurement_noise**2,    # y measurement noise
            (measurement_noise * 0.3)**2  # theta measurement noise
        ])
        
        # Time step
        self.dt = 0.1  # 10 Hz update rate
        
        # Control input
        self.u = np.array([0.0, 0.0])  # [acceleration, steering_rate]
        
    def predict(self, dt: Optional[float] = None,
                control_input: Optional[Tuple[float, float]] = None):
        """
        Predict next state using bicycle model
        
        Args:
            dt: Time step (uses default if None)
            control_input: (acceleration, steering_rate) tuple
        """
        if dt is not None:
            self.dt = dt
        
        if control_input is not None:
            self.u = np.array(control_input)
        
        # State extraction
        x, y, theta, v, omega = self.x
        
        # Bicycle model prediction
        # For simplicity, assume constant velocity and turn rate
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        new_theta = theta + omega * self.dt
        new_v = v + self.u[0] * self.dt  # Apply acceleration
        new_omega = omega + self.u[1] * self.dt  # Apply steering rate
        
        # State transition function
        def f(x_state):
            x_s, y_s, theta_s, v_s, omega_s = x_state
            return np.array([
                x_s + v_s * np.cos(theta_s) * self.dt,
                y_s + v_s * np.sin(theta_s) * self.dt,
                theta_s + omega_s * self.dt,
                v_s,
                omega_s
            ])
        
        # Jacobian of state transition function
        F = np.array([
            [1, 0, -v * np.sin(theta) * self.dt, np.cos(theta) * self.dt, 0],
            [0, 1, v * np.cos(theta) * self.dt, np.sin(theta) * self.dt, 0],
            [0, 0, 1, 0, self.dt],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        
        # Update state
        self.x = np.array([new_x, new_y, new_theta, new_v, new_omega])
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
        
        print(f"[EKF] PREDICT: pos=({new_x:.2f}, {new_y:.2f}), θ={np.degrees(new_theta):.1f}°, v={new_v:.2f}m/s, ω={new_omega:.2f}rad/s")
        
    def update(self, measurement: np.ndarray, 
               H: Optional[np.ndarray] = None):
        """
        Update state with measurement
        
        Args:
            measurement: Measurement vector [x, y, theta]
            H: Measurement matrix (defaults to identity for direct observation)
        """
        if H is None:
            H = np.array([
                [1, 0, 0, 0, 0],  # x
                [0, 1, 0, 0, 0],  # y
                [0, 0, 1, 0, 0]   # theta
            ])
        
        # Predicted measurement
        z_pred = H @ self.x
        
        # Innovation (measurement residual)
        y_innov = measurement - z_pred
        
        # Normalize angle difference
        if len(y_innov) > 2:
            y_innov[2] = self._normalize_angle(y_innov[2])
        
        print(f"[EKF] UPDATE: measurement=({measurement[0]:.2f}, {measurement[1]:.2f}, {np.degrees(measurement[2]):.1f}°)")
        print(f"[EKF]         innovation=({y_innov[0]:.3f}, {y_innov[1]:.3f}, {np.degrees(y_innov[2]):.2f}°)")
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y_innov
        
        # Normalize theta
        self.x[2] = self._normalize_angle(self.x[2])
        
        # Update covariance
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P
        
        print(f"[EKF]         corrected pos=({self.x[0]:.2f}, {self.x[1]:.2f}), θ={np.degrees(self.x[2]):.1f}°")
        
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_state(self) -> VehicleState:
        """Get current state estimate"""
        return VehicleState.from_array(self.x)
    
    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix"""
        return self.P
    
    def reset(self, initial_state: Optional[VehicleState] = None):
        """Reset filter to initial state"""
        if initial_state is None:
            self.x = np.zeros(5)
        else:
            self.x = initial_state.to_array()
        self.P = np.eye(5) * 1.0


class TagMeasurementFusion:
    """
    Fuses AprilTag measurements into Kalman filter
    Handles multiple tag observations and coordinate transformations
    """
    
    def __init__(self, ekf: ExtendedKalmanFilter,
                 tag_map: Optional[dict] = None):
        """
        Initialize measurement fusion
        
        Args:
            ekf: ExtendedKalmanFilter instance
            tag_map: Dictionary mapping tag_id -> (x, y, theta) in world frame
        """
        self.ekf = ekf
        self.tag_map = tag_map or {}
        
    def add_tag_to_map(self, tag_id: int, x: float, y: float, theta: float = 0):
        """Add known tag position to map"""
        self.tag_map[tag_id] = (x, y, theta)
        
    def process_tag_detection(self, tag_detection) -> Optional[np.ndarray]:
        """
        Convert AprilTag detection to world-frame measurement
        
        Args:
            tag_detection: AprilTagDetection object
            
        Returns:
            World-frame measurement [x, y, theta] or None if tag not in map
        """
        tag_id = tag_detection.tag_id
        
        if tag_id not in self.tag_map:
            return None
        
        # Get tag position in world frame
        tag_world_x, tag_world_y, tag_world_theta = self.tag_map[tag_id]
        
        # Get tag pose relative to camera
        tag_pose = tag_detection.pose  # 4x4 matrix, camera to tag
        
        # Extract camera-to-tag translation
        tvec = tag_pose[:3, 3]  # Tag position in camera frame
        
        # Get current vehicle state
        state = self.ekf.get_state()
        
        # Transform tag position from camera frame to world frame
        # Camera is assumed to be at vehicle position, facing forward
        camera_x = state.x
        camera_y = state.y
        camera_theta = state.theta
        
        # Rotate tag position by camera orientation
        tag_world_observed_x = (camera_x + tvec[2] * np.cos(camera_theta) - 
                               tvec[0] * np.sin(camera_theta))
        tag_world_observed_y = (camera_y + tvec[2] * np.sin(camera_theta) + 
                               tvec[0] * np.cos(camera_theta))
        
        # The vehicle position is the negative of the tag-to-vehicle vector
        # Simplified: estimate vehicle position from tag observation
        vehicle_x_estimate = tag_world_x - tvec[2] * np.cos(camera_theta)
        vehicle_y_estimate = tag_world_y - tvec[2] * np.sin(camera_theta)
        
        # Calculate bearing to tag for heading estimate
        bearing = tag_detection.bearing
        vehicle_theta_estimate = tag_world_theta - bearing
        
        # Force all values to pure Python scalars to prevent numpy shape mismatch
        return np.array([
            float(np.asarray(vehicle_x_estimate).item()),
            float(np.asarray(vehicle_y_estimate).item()),
            float(np.asarray(vehicle_theta_estimate).item())
        ])
    
    def fuse_multiple_tags(self, tag_detections: list) -> Optional[np.ndarray]:
        """
        Fuse measurements from multiple tags
        
        Args:
            tag_detections: List of AprilTagDetection objects
            
        Returns:
            Fused measurement [x, y, theta] or None
        """
        measurements = []
        weights = []
        
        for det in tag_detections:
            meas = self.process_tag_detection(det)
            if meas is not None:
                measurements.append(meas)
                # Weight by inverse distance (closer tags are more accurate)
                weight = 1.0 / (det.distance + 0.1)
                weights.append(weight)
        
        if not measurements:
            return None
        
        # Weighted average
        weights = np.array(weights)
        weights /= np.sum(weights)
        
        fused_measurement = np.zeros(3)
        for meas, weight in zip(measurements, weights):
            fused_measurement += weight * meas
        
        return fused_measurement
    
    def update_ekf_with_tags(self, tag_detections: list):
        """
        Update EKF with all available tag measurements
        
        Args:
            tag_detections: List of AprilTagDetection objects
        """
        print(f"[TAG_FUSION] Processing {len(tag_detections)} tag(s) for EKF update...")
        fused_measurement = self.fuse_multiple_tags(tag_detections)
        
        if fused_measurement is not None:
            print(f"[TAG_FUSION] Fused measurement: ({fused_measurement[0]:.2f}, {fused_measurement[1]:.2f}, {np.degrees(fused_measurement[2]):.1f}°)")
            self.ekf.update(fused_measurement)
            print("[TAG_FUSION] EKF update successful\n")
            return True
        else:
            print("[TAG_FUSION] No valid tag measurements to fuse - skipping EKF update\n")
            return False


if __name__ == "__main__":
    # Example usage
    print("Testing Kalman Filter...")
    
    # Initialize with vehicle at origin
    initial_state = VehicleState(x=0, y=0, theta=0, v=0, omega=0)
    ekf = ExtendedKalmanFilter(initial_state=initial_state)
    
    # Create measurement fusion
    fusion = TagMeasurementFusion(ekf)
    
    # Add some known tags to map
    fusion.add_tag_to_map(0, x=5.0, y=10.0, theta=0)
    fusion.add_tag_to_map(1, x=10.0, y=5.0, theta=np.pi/2)
    
    # Simulate some prediction-update cycles
    for i in range(10):
        # Predict (simulate moving forward at 1 m/s)
        ekf.predict(dt=0.1, control_input=(0.0, 0.0))
        
        state = ekf.get_state()
        print(f"Step {i+1}: Position=({state.x:.2f}, {state.y:.2f}), "
              f"Heading={np.degrees(state.theta):.1f}°, Velocity={state.v:.2f} m/s")
    
    print("\nEKF initialized and ready for sensor fusion.")
