# VisCarPath - Autonomous Car Navigation with AprilTags

Autonomous navigation system for routing a car to AprilTag targets using OAK-D Lite stereo camera, with ground plane detection, obstacle avoidance, Kalman filtering, and MPC control.

## Architecture

The system consists of four main modules:

1. **AprilTag Detection** (`apriltag_detection.py`)
   - OAK-D Lite RGB + Depth pipeline
   - AprilTag recognition and pose estimation
   - Ground-level tag filtering

2. **Ground & Obstacle Detection** (`ground_obstacle_detection.py`)
   - RANSAC-based ground plane detection
   - Obstacle detection via height analysis
   - Path planning with obstacle avoidance

3. **Kalman Filter** (`kalman_filter.py`)
   - Extended Kalman Filter for state estimation
   - Vehicle position, velocity, and heading estimation
   - Multi-tag measurement fusion

4. **MPC Controller** (`mpc_controller.py`)
   - Model Predictive Control for path following
   - Bicycle model vehicle dynamics
   - Obstacle-aware trajectory optimization

5. **Main Navigation** (`main_navigation.py`)
   - Integrates all components
   - Real-time control loop
   - Visualization and diagnostics

## Installation

```bash
# Install dependencies
pip install depthai apriltag opencv-python numpy cvxpy

# Optional: for visualization
pip install matplotlib
```

## Usage

### Basic Navigation (with visualization)

```bash
python main_navigation.py
```

### Navigate to Specific Tag

```bash
python main_navigation.py --target 0
```

### Without Visualization (headless mode)

```bash
python main_navigation.py --no-display
```

### Command Line Options

```
--target TAG_ID       Target AprilTag ID to navigate to
--robot-width WIDTH   Robot width in meters (default: 0.5)
--no-display          Disable visualization
```

## Module Details

### AprilTag Detection

```python
from apriltag_detection import OakDAprilTagPipeline

pipeline = OakDAprilTagPipeline(tag_family="tag36h11")
pipeline.start()

rgb, depth, ts = pipeline.get_frame_data()
tags = pipeline.detect_tags_in_frame(rgb, depth)

for tag in tags:
    print(f"Tag {tag.tag_id}: distance={tag.distance:.2f}m, bearing={tag.bearing:.2f}rad")

pipeline.stop()
```

### Ground Plane & Obstacle Detection

```python
from ground_obstacle_detection import GroundAndObstaclePipeline

detector = GroundAndObstaclePipeline(robot_width=0.5)
detector.set_camera_intrinsics(fx, fy, cx, cy)

result = detector.process_frame(depth_map, tag_detections, image_shape)

print(f"Ground confidence: {result['ground_plane'].confidence}")
print(f"Obstacles detected: {len(result['obstacles'])}")
```

### Kalman Filter State Estimation

```python
from kalman_filter import ExtendedKalmanFilter, VehicleState, TagMeasurementFusion

ekf = ExtendedKalmanFilter(initial_state=VehicleState(0, 0, 0, 0, 0))
fusion = TagMeasurementFusion(ekf)

# Add known tag positions
fusion.add_tag_to_map(tag_id=0, x=5.0, y=10.0)

# Update with detections
fusion.update_ekf_with_tags(tag_detections)

# Get estimated state
state = ekf.get_state()
print(f"Position: ({state.x}, {state.y}), Heading: {state.theta}")
```

### MPC Control

```python
from mpc_controller import PathFollowingController, MPCConfig

config = MPCConfig(horizon=10, dt=0.1, max_velocity=1.5)
controller = PathFollowingController(config)

controller.update_state(current_state_array)
acceleration, steering_rate = controller.compute_control(path_segments, obstacles)
```

## System Flow

```
┌─────────────────┐
│   OAK-D Lite    │
│  RGB + Depth    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AprilTag Detect│
│  + Pose Est     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ground Plane    │
│ + Obstacle Det  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Path Planning  │
│  (collision-free)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Kalman Filter   │
│ State Estimate  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MPC Controller  │
│ Optimal Control │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vehicle Control │
│ (accel, steer)  │
└─────────────────┘
```

## Navigation States

- `IDLE` - System initialized, not running
- `DETECTING_TAGS` - Searching for AprilTags
- `PLANNING_PATH` - Computing path to target
- `NAVIGATING` - Actively following path
- `OBSTRUCTED` - Obstacle detected, slowing/stopping
- `TARGET_REACHED` - Arrived at target tag
- `ERROR` - System error

## Requirements

- OAK-D Lite camera
- AprilTag markers (tag36h11 family recommended)
- Python 3.8+
- Dependencies listed above

## Configuration

Key parameters to tune:

- `robot_width`: Your vehicle's width (for collision avoidance)
- `max_velocity`: Maximum forward speed (m/s)
- `obstacle_safety_margin`: Clearance from obstacles (m)
- `camera_pitch`: Camera mounting angle (radians)

## Notes

1. **Camera Mounting**: For best results, mount the OAK-D Lite with a downward pitch of ~15-20 degrees to see both the ground and distant tags.

2. **AprilTag Placement**: Place tags flat on the ground. The system filters for ground-level tags based on orientation.

3. **Lighting**: AprilTag detection works best in good lighting conditions.

4. **Initial Position**: The system starts at origin (0,0). For accurate localization, either:
   - Pre-populate the tag map with known positions
   - Start near a known tag for initial localization

## License

MIT License