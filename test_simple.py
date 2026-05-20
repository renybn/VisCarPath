"""Simple test without depthai dependency"""
import numpy as np
import cv2

print("Testing basic imports...")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Test basic OpenCV functionality
img = np.zeros((100, 100, 3), dtype=np.uint8)
print(f"Created test image: {img.shape}")

# Test apriltag
try:
    import apriltag
    detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
    print("AprilTag detector created successfully")
    
    # Create a simple test image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detector.detect(gray)
    print(f"Detection result: {len(result)} tags found")
    print("✓ All basic tests passed!")
except Exception as e:
    print(f"✗ AprilTag test failed: {e}")
