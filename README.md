# VisCarPath - Autonomous RC Car Navigation

Autonomous navigation system for RC cars using OAK-D Lite camera, AprilTag detection, and model predictive control.

## Installation

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The default `requirements.txt` includes `opencv-python` for GUI support. For headless/server environments, edit `requirements.txt` and replace `opencv-python` with `opencv-python-headless` before installing.

## Running the Program

### Main Navigation Script

Use `main_navigation.py` to run the autonomous navigation system.

#### Basic Usage

```bash
# Navigate to AprilTag ID 0 in visual mode (with display)
python main_navigation.py --target 0 --visual

# Navigate to AprilTag ID 0 in headless mode (saves debug images)
python main_navigation.py --target 0 --headless

# Run bare control loop (no display, no logging)
python main_navigation.py --target 0
```

#### Command Line Options

| Option | Description |
|--------|-------------|
| `--target <id>` | Target AprilTag ID to navigate to |
| `--visual` | Show live RGB + depth windows with obstacle overlay |
| `--headless` | Save debug images to disk without display |
| `--robot-width <m>` | Robot width in meters (default: 0.5) |
| `--log-dir <path>` | Directory for headless debug images (default: nav_logs) |
| `--fastsam-model <path>` | Path to FastSAM weights file (default: FastSAM-s.pt) |

### Debug Script

Use `debug_oakd_comprehensive.py` for detailed OAK-D diagnostics and testing:

```bash
# Run comprehensive OAK-D debugging suite
python debug_oakd_comprehensive.py
```

This script provides performance-optimized depth map validation, AprilTag detection testing, and spatial obstacle detection analysis.

## Project Structure

| File | Description |
|------|-------------|
| `main_navigation.py` | Main navigation pipeline integrating perception, state estimation, and control. Supports visual, headless, and bare modes. |
| `apriltag_detection.py` | AprilTag detection and pose estimation using OAK-D camera. Filters ground-level vs wall-mounted tags. |
| `ground_obstacle_detection.py` | Ground plane detection, obstacle identification, and navigable path mapping using depth data and FastSAM segmentation. |
| `kalman_filter.py` | Extended Kalman Filter for vehicle state estimation (position, velocity, orientation) using bicycle motion model. |
| `mpc_controller.py` | Lightweight geometric path controller with Pure Pursuit and P-Control for obstacle avoidance and path following. |
| `debug_oakd_comprehensive.py` | Comprehensive OAK-D debugging and diagnostic suite with performance optimizations. |
| `requirements.txt` | Python dependencies including OpenCV, DepthAI, AprilTag, and scientific computing libraries. |

## Requirements

- OAK-D Lite camera (optional - system runs in simulation mode without hardware)
- Python 3.8+
- AprilTag markers (tag36h11 family recommended for navigation targets)

## License

MIT License
