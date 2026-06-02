# OAK-D Lite Debugging Suite - Documentation

## Overview

This comprehensive debugging suite helps diagnose and fix common OAK-D Lite issues:
- **Poor AprilTag detection at distance**
- **Noisy/invalid depth maps**
- **False obstacle detections**
- **Depth alignment problems**

The suite runs in **headless mode** by default, outputting detailed metrics to console and saving debug images.

## Files Created

1. **`debug_oakd_comprehensive.py`** - Main debugging script
2. **`debug_frame_XXXX.png`** - Saved debug visualizations (auto-generated)

## Key Features

### 1. Depth Map Analysis (`DepthMapDebugger`)

Analyzes depth quality metrics:
- **Valid pixel percentage** - How much of the scene has valid depth
- **Median depth** - Typical distance to objects
- **Standard deviation** - Depth variation (high = noisy)
- **Noise ratio** - StdDev/Mean (target: <0.3)
- **Temporal stability** - Frame-to-frame consistency (1.0 = perfect)

Visualizations:
- Raw depth map (colorized)
- Filtered depth map (bilateral filter + histogram equalization)
- Statistics overlay

### 2. AprilTag Distance Testing (`AprilTagDistanceTester`)

Tests tag detection with diagnostics:
- Detection confidence (decision margin)
- Apparent tag area (pixels²) - correlates with distance
- 3D position via PnP
- Bearing angle

**Optimization tips:**
- Use `quad_decimate=1.0` for maximum range (full resolution)
- Increase `quad_sigma` slightly (0.0-0.5) for noisy images
- Ensure good lighting and minimal motion blur

### 3. Stereo Depth Configuration (`StereoDepthConfigurator`)

Four preset modes for different scenarios:

| Mode | Best For | Settings |
|------|----------|----------|
| `high_accuracy` | Close-range precision (<2m) | HIGH_ACCURACY preset, extended disparity |
| `balanced` | General use | DEFAULT preset, extended disparity, conf=200 |
| `long_range` | Distant objects (>3m) | LONG_RANGE preset, extended disparity |
| `low_noise` | Noisy environments | HIGH_DENSITY preset, conf=220 |

**Key settings explained:**
- `ExtendedDisparity(True)` - Increases depth range but uses more bandwidth
- `ConfidenceThreshold(200-240)` - Higher = fewer false positives but more holes
- `Subpixel(False)` - Reduces noise at cost of precision

### 4. Spatial Obstacle Detection (`SpatialObstacleDetector`)

Detects obstacles using depth gradients:
- Computes Sobel gradients on depth map
- Finds depth discontinuities (edges)
- Converts to 3D camera coordinates
- Returns position, distance, size

## Usage

### Basic Run (Headless)

```bash
python debug_oakd_comprehensive.py
```

Runs for 300 frames (~10-20 seconds), printing diagnostics every 10 frames and saving debug images every 60 frames.

### Expected Output

```
============================================================
[Frame 10] FPS: 10.4 | Mode: balanced
============================================================
DEPTH MAP ANALYSIS:
  Valid pixels: 85.3%
  Median depth: 1229mm (1.23m)
  StdDev: 416.9mm
  Noise ratio: 0.34 (lower=better, <0.3 good)
  Temporal stability: 0.95 (1.0=perfect)

APRILTAG DETECTIONS: 2 found
  Tag ID 0:
    Distance: 1.45m
    Bearing: -5.2°
    Confidence: 0.87
    Apparent area: 2340 px²

OBSTACLES DETECTED: 3 found
  #1: 0.85m @ (-0.12, 0.45, 0.72)m
  #2: 1.20m @ (0.34, 0.28, 1.10)m
  #3: 1.85m @ (0.05, 0.15, 1.84)m

📸 Saved debug image: debug_frame_0060.png
```

## Troubleshooting Guide

### Problem: Low Valid Pixel Percentage (<50%)

**Causes:**
- Textureless surfaces (white walls, shiny floors)
- Poor lighting
- Reflective surfaces
- USB bandwidth issues

**Solutions:**
1. Add texture to scene (patterned floor, etc.)
2. Improve lighting (avoid direct sunlight)
3. Try `low_noise` mode
4. Check USB connection (use USB 3.0 port/cable)

### Problem: High Noise Ratio (>0.5)

**Causes:**
- Mixed depth values in scene
- Incorrect stereo configuration
- Motion blur

**Solutions:**
1. Use `HIGH_ACCURACY` or `low_noise` preset
2. Increase `confidenceThreshold` to 220-240
3. Reduce camera motion
4. Ensure static scene during testing

### Problem: AprilTags Not Detected at Distance

**Causes:**
- Tag too small in image (<20x20 pixels)
- Motion blur
- Poor lighting
- Wrong focus

**Solutions:**
1. Use larger tags (8cm minimum for >2m range)
2. Set `quad_decimate=1.0` (full resolution)
3. Lock autofocus: `cam_rgb.initialControl.setManualFocus(130)`
4. Improve lighting, reduce exposure time
5. Check apparent tag area in diagnostics

**Rule of thumb:** Tag needs ~400-900 px² apparent area for reliable detection

### Problem: False Obstacle Detections

**Causes:**
- Depth noise creating artificial edges
- Shadows misinterpreted as obstacles
- Ground plane variations

**Solutions:**
1. Increase obstacle detection threshold
2. Apply bilateral filtering to depth
3. Use ground plane subtraction (see `ground_obstacle_detection.py`)
4. Combine with confidence map filtering

### Problem: Depth Values Vary Wildly

**Causes:**
- Autofocus hunting
- Stereo rectification errors
- USB bandwidth saturation

**Solutions:**
1. **Lock autofocus** - Critical! Use manual focus or CONTINUOUS_VIDEO
2. Check calibration: `device.readCalibration()`
3. Reduce resolution or FPS
4. Use USB 3.0 cable/port

## Integration with Existing Code

### Update `apriltag_detection.py`

Already updated with optimized settings:
```python
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setExtendedDisparity(True)
stereo.setConfidenceThreshold(220)
```

### Using Depth Debugger in Your Code

```python
from debug_oakd_comprehensive import DepthMapDebugger

debugger = DepthMapDebugger()

# In your main loop
depth_stats = debugger.analyze_depth_map(depth_frame)
print(f"Depth quality: {depth_stats['valid_pixels']/depth_stats['total_pixels']*100:.1f}%")
print(f"Noise ratio: {depth_stats['noise_ratio']:.2f}")

if depth_stats['noise_ratio'] > 0.5:
    print("WARNING: High depth noise detected!")
```

### Using AprilTag Tester

```python
from debug_oakd_comprehensive import AprilTagDistanceTester

tester = AprilTagDistanceTester()

# In your main loop
tag_detections = tester.detect_with_diagnostics(gray_frame, K_matrix)

for tag in tag_detections:
    if tag['apparent_area_px'] < 400:
        print(f"Warning: Tag {tag['tag_id']} is small ({tag['apparent_area_px']:.0f}px²)")
    if tag['confidence'] < 0.5:
        print(f"Warning: Low confidence ({tag['confidence']:.2f})")
```

## Debug Image Format

Saved images (`debug_frame_XXXX.png`) show:
- **Left panel**: RGB frame with tag/obstacle overlays
- **Right panel**: Colorized depth map
- Green boxes: Detected AprilTags with ID and distance
- Colored dots: Obstacles (red=close, green=far)

## Performance Benchmarks

Expected performance on typical hardware:

| Scenario | FPS | Notes |
|----------|-----|-------|
| Simulation mode | 10-15 | CPU-bound (synthetic data) |
| OAK-D Lite USB 3.0 | 12-18 | Balanced mode, 640x480 |
| OAK-D Lite USB 2.0 | 8-12 | Bandwidth limited |

## Next Steps

1. **Run the debugger** with your actual setup
2. **Review debug images** for visual confirmation
3. **Adjust parameters** based on metrics
4. **Integrate fixes** into main navigation code
5. **Test with real tags** at various distances

## Additional Resources

- [Luxonis OAK-D Lite Docs](https://docs.luxonis.com/hardware/products/OAK-D%20Lite)
- [StereoDepth Node API](https://docs.luxonis.com/api/nodes/StereoDepth/)
- [AprilTag Best Practices](https://april.eecs.umich.edu/software/apriltag.html)
