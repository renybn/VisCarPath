"""
Threaded IMU Integration for Visual-Inertial Odometry
Decouples 100Hz IMU updates from USB 2.0 camera frame rate.
Uses BNO085 Game Rotation Vector or BMI270 Gyroscope.
"""
import threading
import time
import numpy as np
from typing import Optional
from queue import Queue, Empty
import depthai as dai


class ThreadedIMU:
    """
    Background IMU processor that maintains a thread-safe vehicle state.
    Provides yaw rate and orientation independent of camera latency.
    """

    def __init__(self, use_rotation_vector: bool = True):
        self.use_rotation_vector = use_rotation_vector
        self._state_lock = threading.Lock()

        # Thread-safe state: [yaw (rad), yaw_rate (rad/s), timestamp]
        self._state = np.array([0.0, 0.0, 0.0])
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: Optional[dai.DataOutputQueue] = None

    def start(self, device: dai.Device):
        """Start IMU background thread with existing DepthAI device."""
        if self._running:
            return

        self._queue = device.getOutputQueue(name="imu", maxSize=30, blocking=False)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop IMU background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_state(self) -> np.ndarray:
        """
        Get latest IMU state (thread-safe copy).
        Returns: [yaw (rad), yaw_rate (rad/s), timestamp (s)]
        """
        with self._state_lock:
            return self._state.copy()

    def _run_loop(self):
        """Background loop: read batched IMU packets and integrate yaw."""
        prev_yaw = 0.0
        prev_time = None

        while self._running:
            try:
                imu_data = self._queue.get(timeout=0.05)
            except Empty:
                continue

            packets = imu_data.packets
            for packet in packets:
                ts = packet.timestamp.total_seconds()

                if self.use_rotation_vector:
                    # BNO085/BNO086: onboard sensor fusion quaternion
                    rv = packet.rotationVector
                    # Convert quaternion to yaw (Z-axis rotation)
                    yaw = np.arctan2(
                        2.0 * (rv.w * rv.z + rv.x * rv.y),
                        1.0 - 2.0 * (rv.y * rv.y + rv.z * rv.z)
                    )
                else:
                    # BMI270 fallback: raw calibrated gyroscope Z-axis
                    gyro = packet.gyroscope
                    yaw_rate = gyro.z
                    if prev_time is not None:
                        dt = ts - prev_time
                        yaw = prev_yaw + yaw_rate * dt
                    else:
                        yaw = prev_yaw

                # Compute yaw rate from successive yaw measurements
                if prev_time is not None:
                    dt = ts - prev_time
                    if dt > 1e-6:
                        dyaw = yaw - prev_yaw
                        # Handle angle wrapping for rate computation
                        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
                        yaw_rate = dyaw / dt
                    else:
                        yaw_rate = 0.0
                else:
                    yaw_rate = 0.0

                prev_yaw = yaw
                prev_time = ts

                # Atomic state update
                with self._state_lock:
                    self._state[:] = [yaw, yaw_rate, ts]

    @staticmethod
    def create_imu_node(pipeline: dai.Pipeline,
                        use_rotation_vector: bool = True) -> dai.node.IMU:
        """
        Create and configure IMU node optimized for USB 2.0 bandwidth.
        Call this BEFORE starting the DepthAI device.
        """
        imu = pipeline.create(dai.node.IMU)

        if use_rotation_vector:
            # BNO085/086: onboard fusion at 100Hz
            imu.enableIMUSensor([dai.IMUSensor.GAME_ROTATION_VECTOR], 100)
        else:
            # BMI270: raw gyro + accel at 100Hz
            imu.enableIMUSensor([
                dai.IMUSensor.GYROSCOPE_CALIBRATED,
                dai.IMUSensor.ACCELEROMETER
            ], 100)

        # CRITICAL FOR USB 2.0: batch packets to reduce interrupt overhead
        # At 100Hz with batch=5, host receives ~20 batches/sec instead of 100
        imu.setBatchReportThreshold(5)
        imu.setMaxBatchReports(10)

        xout_imu = pipeline.create(dai.node.XLinkOut)
        xout_imu.setStreamName("imu")
        imu.out.link(xout_imu.input)

        return imu