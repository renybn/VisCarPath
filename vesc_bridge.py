"""
VESC Bridge - Direct Serial Implementation
No pyvesc dependency - uses raw VESC protocol over pyserial only
"""

import serial
import struct

def _crc_ccitt(data: bytes) -> int:
    """VESC uses CRC-CCITT for packet integrity"""
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
        crc &= 0xFFFF
    return crc

def _build_packet(payload: bytes) -> bytes:
    """Wrap payload in VESC packet format"""
    length = len(payload)
    header = bytes([0x02, length])
    crc    = _crc_ccitt(payload)
    footer = bytes([crc >> 8, crc & 0xFF, 0x03])
    return header + payload + footer

def _pack_rpm(rpm: int) -> bytes:
    """VESC command 8 — Set motor RPM"""
    payload = bytes([8]) + struct.pack('>i', int(rpm))
    return _build_packet(payload)

def _pack_servo(position: float) -> bytes:
    """VESC command 23 — Set servo position (0.0 to 1.0)"""
    payload = bytes([23]) + struct.pack('>f', float(position))
    return _build_packet(payload)

def _pack_duty(duty: float) -> bytes:
    """VESC command 5 — Set duty cycle (-1.0 to 1.0)"""
    payload = bytes([5]) + struct.pack('>i', int(duty * 100000))
    return _build_packet(payload)


class VESCBridge:
    def __init__(self,
                 port: str = '/dev/ttyACM0',
                 baud_rate: int = 115200,
                 max_erpm: float = 3000.0,
                 max_accel: float = 0.8,
                 max_steer_rate: float = 2.0):

        self.max_erpm      = max_erpm
        self.max_accel     = max_accel
        self.max_steer_rate = max_steer_rate
        self.servo_center  = 0.5
        self.servo_range   = 0.3

        try:
            self.serial = serial.Serial(port, baud_rate, timeout=0.1)
            print(f"[VESC] Connected on {port}")
        except serial.SerialException as e:
            print(f"[VESC] Connection failed: {e}")
            self.serial = None

    def _normalize_accel(self, accel: float) -> float:
        return float(accel) / self.max_accel

    def _normalize_steer(self, steer_rate: float) -> float:
        return float(steer_rate) / self.max_steer_rate

    def _accel_to_erpm(self, accel_normalized: float) -> int:
        return int(accel_normalized * self.max_erpm)

    def _steer_to_servo(self, steer_normalized: float) -> float:
        servo = self.servo_center + steer_normalized * self.servo_range
        return float(max(0.0, min(1.0, servo)))

    def send_command(self, accel: float, steer_rate: float):
        if self.serial is None:
            print("[VESC] No connection — skipping command")
            return

        accel_norm = self._normalize_accel(accel)
        steer_norm = self._normalize_steer(steer_rate)
        erpm       = self._accel_to_erpm(accel_norm)
        servo      = self._steer_to_servo(steer_norm)

        try:
            self.serial.write(_pack_servo(servo))
            self.serial.write(_pack_rpm(erpm))
            print(f"[VESC] ERPM: {erpm:+6d} | Servo: {servo:.3f} "
                  f"(accel={accel:+.2f}, steer={steer_rate:+.2f})")
        except Exception as e:
            print(f"[VESC] Send error: {e}")

    def stop(self):
        """Emergency stop"""
        if self.serial:
            try:
                self.serial.write(_pack_servo(self.servo_center))
                self.serial.write(_pack_rpm(0))
                print("[VESC] Emergency stop sent")
            except Exception as e:
                print(f"[VESC] Stop error: {e}")

    def close(self):
        """Clean shutdown"""
        self.stop()
        if self.serial:
            self.serial.close()
            print("[VESC] Connection closed")