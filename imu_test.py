from imu_integration import ThreadedIMU
import depthai as dai
import time

# Create minimal pipeline with ONLY IMU (no cameras)
pipeline = dai.Pipeline()
ThreadedIMU.create_imu_node(pipeline, use_rotation_vector=True)

with dai.Device(pipeline, usb2Mode=True) as device:
    imu = ThreadedIMU(use_rotation_vector=True)
    imu.start(device)
    
    start = time.time()
    count = 0
    while time.time() - start < 5.0:
        state = imu.get_state()
        if state[2] > 0:  # Valid timestamp
            count += 1
        time.sleep(0.001)
    
    imu.stop()
    print(f"Received {count} updates in 5s ({count/5:.1f} Hz)")
    # Expected: ~480-500 (100Hz * 5s, minus startup)