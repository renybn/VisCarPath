"""
MPC (Model Predictive Control) for Autonomous Vehicle Path Following
Computes optimal control inputs to follow planned path while avoiding obstacles
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import cvxpy as cp


@dataclass
class MPCConfig:
    """MPC controller configuration"""
    horizon: int = 10  # Prediction horizon steps
    dt: float = 0.1  # Time step (seconds)
    
    # Vehicle constraints
    max_velocity: float = 2.0  # m/s
    max_acceleration: float = 1.0  # m/s²
    max_steering_rate: float = 1.0  # rad/s
    max_steering_angle: float = 0.5  # rad
    
    # Cost weights
    weight_position: float = 10.0
    weight_heading: float = 5.0
    weight_velocity: float = 1.0
    weight_acceleration: float = 0.1
    weight_steering_rate: float = 0.1
    
    # Obstacle avoidance
    obstacle_safety_margin: float = 0.3  # meters
    obstacle_weight: float = 100.0


@dataclass
class MPCResult:
    """MPC optimization result"""
    success: bool
    acceleration: float  # Optimal acceleration command
    steering_rate: float  # Optimal steering rate command
    predicted_trajectory: List[Tuple[float, float, float]]  # [(x, y, theta), ...]
    cost: float  # Final cost value
    solve_time: float  # Solver time in seconds


class VehicleDynamics:
    """
    Bicycle model vehicle dynamics
    """
    
    def __init__(self, wheelbase: float = 0.5):
        """
        Initialize vehicle dynamics
        
        Args:
            wheelbase: Distance between front and rear axles (meters)
        """
        self.L = wheelbase  # Wheelbase
        
    def kinematics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Compute state derivatives
        
        Args:
            state: [x, y, theta, v, delta] (position, heading, velocity, steering angle)
            control: [a, delta_dot] (acceleration, steering rate)
            
        Returns:
            State derivatives [dx, dy, dtheta, dv, ddelta]
        """
        x, y, theta, v, delta = state
        a, delta_dot = control
        
        # Bicycle model kinematics
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * np.tan(delta) / self.L if abs(delta) < 0.49 * np.pi else 0
        dv = a
        ddelta = delta_dot
        
        return np.array([dx, dy, dtheta, dv, ddelta])
    
    def predict_step(self, state: np.ndarray, control: np.ndarray, 
                     dt: float) -> np.ndarray:
        """
        Predict next state using Euler integration
        
        Args:
            state: Current state
            control: Control input
            dt: Time step
            
        Returns:
            Next state
        """
        derivatives = self.kinematics(state, control)
        return state + derivatives * dt


class MPController:
    """
    Nonlinear Model Predictive Controller for path following
    Uses CVXPY for convex optimization (with linearization)
    """
    
    def __init__(self, config: Optional[MPCConfig] = None):
        """
        Initialize MPC controller
        
        Args:
            config: MPC configuration parameters
        """
        self.config = config or MPCConfig()
        self.dynamics = VehicleDynamics()
        
        # State dimension: [x, y, theta, v, delta]
        self.nx = 5
        # Control dimension: [acceleration, steering_rate]
        self.nu = 2
        
    def linearize_dynamics(self, state_nominal: np.ndarray,
                          control_nominal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around nominal trajectory
        
        Args:
            state_nominal: Nominal state
            control_nominal: Nominal control
            
        Returns:
            A, B matrices for linearized system: x_{k+1} = A*x_k + B*u_k
        """
        x, y, theta, v, delta = state_nominal
        dt = self.config.dt
        L = self.dynamics.L
        
        # Jacobian of dynamics w.r.t. state (A matrix)
        tan_delta = np.tan(delta)
        sec_delta_sq = 1.0 / (np.cos(delta) ** 2) if abs(delta) < 0.49 * np.pi else 1.0
        
        A = np.array([
            [1, 0, -v * np.sin(theta) * dt, np.cos(theta) * dt, 0],
            [0, 1, v * np.cos(theta) * dt, np.sin(theta) * dt, 0],
            [0, 0, 1, tan_delta / L * dt, v * sec_delta_sq / L * dt],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        
        # Jacobian of dynamics w.r.t. control (B matrix)
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [dt, 0],
            [0, dt]
        ])
        
        # Affine term
        c = state_nominal - A @ state_nominal
        
        return A, B, c
    
    def solve_mpc(self, current_state: np.ndarray,
                  reference_trajectory: List[np.ndarray],
                  obstacles: Optional[List[dict]] = None) -> MPCResult:
        """
        Solve MPC optimization problem
        
        Args:
            current_state: Current vehicle state [x, y, theta, v, delta]
            reference_trajectory: List of reference states [x, y, theta, v]
            obstacles: List of obstacles with 'position' and 'radius' keys
            
        Returns:
            MPCResult with optimal controls and predicted trajectory
        """
        import time
        start_time = time.time()
        
        N = self.config.horizon
        
        # Ensure reference trajectory matches horizon
        while len(reference_trajectory) < N + 1:
            if reference_trajectory:
                reference_trajectory.append(reference_trajectory[-1].copy())
            else:
                reference_trajectory.append(current_state[:4])
        
        # Decision variables
        X = cp.Variable((self.nx, N + 1))  # States
        U = cp.Variable((self.nu, N))      # Controls
        
        # Cost function
        cost = 0
        
        for k in range(N):
            # State error cost
            x_error = X[:4, k] - reference_trajectory[k][:4]
            cost += self.config.weight_position * cp.sum_squares(x_error[:2])
            cost += self.config.weight_heading * cp.sum_squares(x_error[2])
            cost += self.config.weight_velocity * cp.sum_squares(x_error[3])
            
            # Control effort cost
            cost += self.config.weight_acceleration * cp.sum_squares(U[0, k])
            cost += self.config.weight_steering_rate * cp.sum_squares(U[1, k])
            
            # Obstacle avoidance (soft constraints with barrier function)
            if obstacles:
                for obs in obstacles:
                    obs_pos = obs.get('position', np.zeros(2))
                    obs_radius = obs.get('radius', 0.2)
                    
                    # Distance from vehicle to obstacle
                    dist = cp.norm(X[:2, k] - obs_pos)
                    
                    # Barrier function for safety margin
                    safe_dist = obs_radius + self.config.obstacle_safety_margin
                    cost += self.config.obstacle_weight * cp.pos(safe_dist - dist)**2
        
        # Terminal cost
        x_terminal_error = X[:4, N] - reference_trajectory[N][:4]
        cost += self.config.weight_position * cp.sum_squares(x_terminal_error[:2])
        cost += self.config.weight_heading * cp.sum_squares(x_terminal_error[2])
        
        # Constraints
        constraints = []
        
        # Initial state
        constraints.append(X[:, 0] == current_state)
        
        # Dynamics constraints (linearized)
        for k in range(N):
            # Use simple linear approximation
            state_k = reference_trajectory[k] if k < len(reference_trajectory) else current_state
            control_k = np.zeros(2)
            
            A, B, c = self.linearize_dynamics(state_k, control_k)
            constraints.append(X[:, k+1] == A @ X[:, k] + B @ U[:, k] + c)
        
        # Input constraints
        constraints.append(U[0, :] <= self.config.max_acceleration)
        constraints.append(U[0, :] >= -self.config.max_acceleration)
        constraints.append(U[1, :] <= self.config.max_steering_rate)
        constraints.append(U[1, :] >= -self.config.max_steering_rate)
        
        # State constraints
        constraints.append(X[3, :] <= self.config.max_velocity)
        constraints.append(X[3, :] >= 0)  # No reverse
        constraints.append(cp.abs(X[4, :]) <= self.config.max_steering_angle)
        
        # Obstacle hard constraints (minimum distance)
        if obstacles:
            for obs in obstacles:
                obs_pos = obs.get('position', np.zeros(2))
                obs_radius = obs.get('radius', 0.2)
                
                for k in range(N + 1):
                    dist = cp.norm(X[:2, k] - obs_pos)
                    constraints.append(dist >= obs_radius + self.config.obstacle_safety_margin * 0.5)
        
        # Solve optimization
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=1000)
            solve_time = time.time() - start_time
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                # Extract solution
                optimal_U = U.value
                optimal_X = X.value
                
                # Get first control input
                acceleration = optimal_U[0, 0]
                steering_rate = optimal_U[1, 0]
                
                # Build predicted trajectory
                predicted_trajectory = []
                for k in range(N + 1):
                    predicted_trajectory.append((
                        optimal_X[0, k],
                        optimal_X[1, k],
                        optimal_X[2, k]
                    ))
                
                return MPCResult(
                    success=True,
                    acceleration=acceleration,
                    steering_rate=steering_rate,
                    predicted_trajectory=predicted_trajectory,
                    cost=problem.value,
                    solve_time=solve_time
                )
            else:
                return MPCResult(
                    success=False,
                    acceleration=0.0,
                    steering_rate=0.0,
                    predicted_trajectory=[],
                    cost=float('inf'),
                    solve_time=solve_time
                )
                
        except Exception as e:
            print(f"MPC solver error: {e}")
            return MPCResult(
                success=False,
                acceleration=0.0,
                steering_rate=0.0,
                predicted_trajectory=[],
                cost=float('inf'),
                solve_time=time.time() - start_time
            )
    
    def compute_reference_trajectory(self, path_segments: list,
                                    current_state: np.ndarray) -> List[np.ndarray]:
        """
        Generate reference trajectory from path segments
        
        Args:
            path_segments: List of PathSegment objects
            current_state: Current vehicle state
            
        Returns:
            List of reference states [x, y, theta, v]
        """
        reference = []
        
        if not path_segments:
            # No path, maintain current state
            for _ in range(self.config.horizon + 1):
                reference.append(current_state[:4])
            return reference
        
        # Sample points along path segments
        total_points = self.config.horizon + 1
        points_per_segment = max(1, total_points // len(path_segments))
        
        for segment in path_segments:
            direction = segment.end - segment.start
            segment_length = np.linalg.norm(direction)
            
            if segment_length < 0.01:
                continue
                
            unit_direction = direction / segment_length
            
            for i in range(points_per_segment):
                t = i / points_per_segment
                point = segment.start + t * direction
                
                # Calculate heading from direction
                theta = np.arctan2(unit_direction[1], unit_direction[0])
                
                # Target velocity based on clearance
                v = min(self.config.max_velocity, segment.clearance * 2)
                
                reference.append(np.array([point[0], point[1], theta, v]))
        
        # Pad if needed
        while len(reference) < total_points:
            if reference:
                reference.append(reference[-1].copy())
            else:
                reference.append(current_state[:4])
        
        return reference[:total_points]


class PathFollowingController:
    """
    High-level controller combining MPC with path planning
    """
    
    def __init__(self, mpc_config: Optional[MPCConfig] = None):
        """Initialize path following controller"""
        self.mpc = MPController(mpc_config)
        self.current_state = np.zeros(5)  # [x, y, theta, v, delta]
        
    def update_state(self, state: np.ndarray):
        """Update current vehicle state from Kalman filter"""
        self.current_state = state
        
    def compute_control(self, path_segments: list,
                       obstacles: list) -> Tuple[float, float]:
        """
        Compute optimal control inputs
        
        Args:
            path_segments: Planned path segments
            obstacles: List of obstacles to avoid
            
        Returns:
            (acceleration, steering_rate) commands
        """
        # Generate reference trajectory
        reference = self.mpc.compute_reference_trajectory(
            path_segments, self.current_state
        )
        
        # Format obstacles for MPC
        mpc_obstacles = []
        for obs in obstacles:
            mpc_obstacles.append({
                'position': obs.position_3d[:2],
                'radius': max(obs.size) * 0.5 if hasattr(obs, 'size') else 0.2
            })
        
        # Solve MPC
        result = self.mpc.solve_mpc(
            self.current_state,
            reference,
            mpc_obstacles if mpc_obstacles else None
        )
        
        if result.success:
            return result.acceleration, result.steering_rate
        else:
            # Fallback: simple PD control toward path
            return self._fallback_control(path_segments)
    
    def _fallback_control(self, path_segments: list) -> Tuple[float, float]:
        """Simple fallback control when MPC fails"""
        if not path_segments:
            return 0.0, 0.0
        
        # Go to first segment endpoint
        target = path_segments[0].end
        current_pos = self.current_state[:2]
        
        # Vector to target
        to_target = target - current_pos
        distance = np.linalg.norm(to_target)
        
        if distance < 0.1:
            return 0.0, 0.0
        
        # Desired heading
        desired_theta = np.arctan2(to_target[1], to_target[0])
        heading_error = desired_theta - self.current_state[2]
        
        # Normalize heading error
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Simple P control
        acceleration = min(0.5, distance)  # P control on distance
        steering_rate = 0.5 * heading_error  # P control on heading
        
        return acceleration, steering_rate


if __name__ == "__main__":
    print("Testing MPC Controller...")
    
    # Create controller
    config = MPCConfig(horizon=10, dt=0.1)
    controller = PathFollowingController(config)
    
    # Set initial state
    initial_state = np.array([0, 0, 0, 0, 0])
    controller.update_state(initial_state)
    
    # Create dummy path
    from ground_obstacle_detection import PathSegment
    import numpy as np
    
    path = [
        PathSegment(
            start=np.array([0, 0, 0]),
            end=np.array([2, 0, 0]),
            width=0.5,
            clearance=1.0,
            cost=2.0
        )
    ]
    
    # Compute control
    accel, steer = controller.compute_control(path, [])
    print(f"Control commands: acceleration={accel:.3f} m/s², steering_rate={steer:.3f} rad/s")
    
    print("\nMPC controller initialized and ready.")
