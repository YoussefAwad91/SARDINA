#!/usr/bin/env python3
# =============================================================================
# dummy_ultrasonic_node.py
# Simulates 6 HC-SR04 ultrasonic sensors for testing without hardware.
#
# Each sensor follows an independent sine wave so the dashboard sees a
# naturally changing environment — obstacles approaching and receding.
# Sensors are slightly phase-shifted from each other so they don't all
# move in lockstep, giving a more realistic feel.
#
# Published topics:
#   /ultrasonic/1  through  /ultrasonic/6   std_msgs/Float32  (metres)
#
# Usage:
#   python3 dummy_ultrasonic_node.py
#
# Optional args (ros2 param style, set at launch):
#   publish_rate   float   Hz, default 10.0
#   min_dist       float   metres, default 0.3
#   max_dist       float   metres, default 3.5
# =============================================================================

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# ---------------------------------------------------------------------------
# Simulation profile for each sensor
# Each entry: (amplitude_m, period_s, phase_offset_rad, base_distance_m)
# The simulated distance = base + amplitude * sin(2π/period * t + phase)
# Clamped to [min_dist, max_dist] declared as node params.
# ---------------------------------------------------------------------------
SENSOR_PROFILES = {
    #  idx:  (amplitude, period_s, phase_rad,  base_m)
    1:       (0.6,       6.0,      0.0,         1.5),   # front — slow sweep
    2:       (0.4,       4.0,      1.0,         1.2),   # front-right — medium
    3:       (0.8,       5.0,      2.1,         1.8),   # right — wide swing
    4:       (0.5,       7.0,      3.5,         2.0),   # rear — slow
    5:       (0.7,       3.5,      4.2,         1.4),   # left — fast
    6:       (0.3,       4.5,      5.8,         1.1),   # front-left — tight
}

class DummyUltrasonicNode(Node):

    def __init__(self):
        super().__init__("dummy_ultrasonic_node")

        # --- Declare parameters so they can be overridden at launch ---
        self.declare_parameter("publish_rate", 10.0)   # Hz
        self.declare_parameter("min_dist",     0.02)   # metres (HC-SR04 min)
        self.declare_parameter("max_dist",     4.00)   # metres (HC-SR04 max)

        self._rate    = self.get_parameter("publish_rate").value
        self._min_d   = self.get_parameter("min_dist").value
        self._max_d   = self.get_parameter("max_dist").value

        # --- Create one publisher per sensor ---
        self._ultrasonic_publishers: dict[int, rclpy.publisher.Publisher] = {}
        for idx in SENSOR_PROFILES:
            topic = f"/ultrasonic/_{idx}"
            self._ultrasonic_publishers[idx] = self.create_publisher(Float32, topic, 10)
            self.get_logger().info(f"Publishing dummy data on {topic}")

        # --- Single timer drives all sensors ---
        self._t = 0.0   # simulated time accumulator in seconds
        self._dt = 1.0 / self._rate
        self.create_timer(self._dt, self._tick)

        self.get_logger().info(
            f"Dummy ultrasonic node ready at {self._rate}Hz "
            f"(range {self._min_d}m – {self._max_d}m)"
        )

    # -----------------------------------------------------------------------
    # Timer callback
    # -----------------------------------------------------------------------

    def _tick(self):
        """Advance simulation time and publish one reading per sensor."""
        self._t += self._dt

        for idx, (amplitude, period, phase, base) in SENSOR_PROFILES.items():
            raw = base + amplitude * math.sin((2 * math.pi / period) * self._t + phase)

            # Clamp to valid HC-SR04 range
            raw = max(self._min_d, min(self._max_d, raw))

            msg = Float32()
            msg.data = float(raw)
            self._ultrasonic_publishers[idx].publish(msg)

        self.get_logger().debug(f"[t={self._t:.2f}s] Published all sensor readings")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DummyUltrasonicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()