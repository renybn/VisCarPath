
## Installation

### Quick Install (One Command)

Copy and paste this command to install all required dependencies:

```bash
pip install opencv-python>=4.5.0 apriltag>=3.0.0 depthai<3.0.0 numpy>=1.20.0 scipy>=1.7.0 cvxpy>=1.0.0 osqp>=0.6.0
```

**Important Notes:**
- **DepthAI**: Pinned to `<3.0.0` to avoid breaking API changes in v3.0
- **OpenCV Variants**: 
  - Use `opencv-python>=4.5.0` for local execution (with GUI/display support via `cv2.imshow()`)
  - For cloud/server environments (GitHub Codespaces, Docker), replace with `opencv-python-headless>=4.5.0` to prevent `libGL.so.1` errors
- **Optional Solvers**: `osqp` is recommended for faster MPC optimization

### Alternative: Install from requirements.txt

```bash
# For local execution (default - includes opencv-python with GUI support)
pip install -r requirements.txt

# For cloud/server execution (edit requirements.txt first to use opencv-python-headless)
```

---

## Testing and Debugging

The system includes comprehensive test suites for each module. All tests currently pass (34/34 total).

### Quick Start: Run All Tests

```bash
# Run all test suites sequentially
python test_apriltag.py && python test_ground_obstacle.py && \
python test_kalman_filter.py && python test_mpc_controller.py
```

**Expected Output:**
- AprilTag: 7/7 tests passed ✓
- Ground/Obstacle: 8/8 tests passed ✓
- Kalman Filter: 9/9 tests passed ✓
- MPC Controller: 10/10 tests passed ✓

### Running Individual Test Suites

```bash
# Test AprilTag detection module (7 tests)
python test_apriltag.py

# Test ground plane and obstacle detection (8 tests)
python test_ground_obstacle.py

# Test Kalman filter state estimation (9 tests)
python test_kalman_filter.py

# Test MPC controller (10 tests)
python test_mpc_controller.py
```

### Test Suite Descriptions

#### AprilTag Detection Tests (test_apriltag.py) - 7 Tests
1. **Initialization** - Verify detector creation with default/custom parameters
2. **Camera Intrinsics** - Test setting focal length and principal point
3. **Synthetic Detection** - Detect tags in test images (skipped if no test_tag.png)
4. **Ground Filtering** - Filter ground-level vs wall-mounted tags
5. **Pose Estimation** - Verify PnP accuracy with known geometry
6. **Bearing Calculation** - Test angle calculations for various positions
7. **OAK-D Pipeline** - Verify pipeline structure (without hardware)

#### Ground & Obstacle Detection Tests (test_ground_obstacle.py) - 8 Tests
1. **Ground Detector Init** - Verify RANSAC parameters
2. **Synthetic Ground Detection** - Detect plane in synthetic point cloud
3. **Obstacle Detector Init** - Verify height threshold settings
4. **Synthetic Obstacle Detection** - Detect obstacles in depth data
5. **Path Planner Init** - Verify robot width and clearance settings
6. **Cost Map Creation** - Generate navigation cost maps
7. **Path Segment Creation** - Create and validate path segments
8. **Pipeline Structure** - Test integrated pipeline setup

#### Kalman Filter Tests (test_kalman_filter.py) - 9 Tests
1. **Vehicle State Creation** - Create and convert state objects
2. **EKF Initialization** - Set up filter with custom initial state
3. **EKF Prediction** - Test bicycle model prediction
4. **EKF Update** - Verify measurement correction
5. **Angle Normalization** - Test angle wrapping to [-pi, pi]
6. **Tag Fusion Init** - Initialize measurement fusion
7. **Tag Measurement Processing** - Convert tag detections to measurements
8. **Multiple Tag Fusion** - Combine measurements from multiple tags
9. **EKF Reset** - Reset filter to initial conditions

#### MPC Controller Tests (test_mpc_controller.py) - 10 Tests
1. **MPC Config** - Create configuration with constraints
2. **Vehicle Dynamics** - Test bicycle model kinematics
3. **Steering Kinematics** - Verify turning radius calculations
4. **MPC Controller Init** - Initialize optimization controller
5. **Dynamics Linearization** - Compute Jacobian matrices
6. **Reference Trajectory** - Generate path references
7. **Simple MPC Solve** - Optimize control inputs (no obstacles)
8. **MPC with Obstacles** - Test obstacle avoidance
9. **Path Following Controller** - High-level control interface
10. **Fallback Control** - PD control when MPC fails

### Debugging Tips

**AprilTag Detection Issues:**
- If synthetic detection fails, create a test image with an AprilTag
- Check camera intrinsics match your OAK-D calibration
- Adjust quad_decimate for faster/slower detection
- Ensure apriltag library is installed: `pip install apriltag`

**Ground Plane Detection Issues:**
- Increase min_inliers if too many false positives
- Decrease ransac_threshold for stricter plane fitting
- Ensure depth data has sufficient ground points visible

**Kalman Filter Issues:**
- Increase process_noise if filter is too slow to track
- Decrease measurement_noise to trust measurements more
- Check tag map contains correct world positions

**MPC Controller Issues:**
- Reduce horizon if solve time is too long
- Increase obstacle_safety_margin for more conservative navigation
- Check CVXPY solver installation: `pip install cvxpy osqp`

### Creating Test Data

For testing with real AprilTags:
1. Download AprilTag family images from the [AprilTag GitHub](https://github.com/AprilRobotics/apriltag)
2. Print tag36h11 family tags for best results
3. Place tags flat on the ground for navigation targets
4. Measure and record tag positions for the tag_map in kalman_filter.py

### Integration Testing

To test the full navigation pipeline:

```bash
# Run main navigation (works with or without OAK-D hardware)
python main_navigation.py --target 0 --no-display
```

This will:
1. Initialize OAK-D camera (or run in simulation mode if no hardware)
2. Detect AprilTags and estimate poses
3. Build ground/obstacle map
4. Plan path to target tag
5. Run Kalman filter for state estimation
6. Execute MPC control commands

**Expected Output:**
- System initializes successfully
- Runs in "detecting_tags" state (waiting for tags in simulation mode)
- Outputs diagnostic information every 100ms
- Press Ctrl+C to stop

**With GUI (requires display):**
```bash
python main_navigation.py --target 0
```

**Without OAK-D Hardware:**
The system automatically detects if OAK-D is available and runs in simulation mode if not connected. This allows testing the logic without physical hardware.

**With OAK-D Hardware:**
Connect your OAK-D Lite via USB and run:
```bash
python main_navigation.py --target 0
```

The system will:
- Connect to the OAK-D device
- Stream RGB and depth frames
- Perform real-time AprilTag detection
- Navigate toward detected ground tags

## License

MIT License
