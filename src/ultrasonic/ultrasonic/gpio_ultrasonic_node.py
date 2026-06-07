#!/usr/bin/env python3
# =============================================================================
# gpio_ultrasonic_node.py
# Reads 6 HC-SR04 ultrasonic sensors via GPIO and publishes to /ultrasonic/n.
#
# Uses gpiozero.DistanceSensor which handles the trigger/echo timing
# internally in a background thread — no blocking calls in the ROS loop.
#
# Wiring per sensor (repeat for all 6):
#
#   HC-SR04 VCC  → Pi 5V  (pin 2 or 4)
#   HC-SR04 GND  → Pi GND (pin 6, 9, 14, 20, 25, 30, 34, or 39)
#   HC-SR04 TRIG → Pi GPIO (see SENSOR_GPIO_PINS below)
#   HC-SR04 ECHO → Pi GPIO (see SENSOR_GPIO_PINS below)
#
#   IMPORTANT: The HC-SR04 ECHO pin outputs 5V logic.
#   The Raspberry Pi GPIO is 3.3V tolerant only.
#   Use a voltage divider on the ECHO line:
#     ECHO → 1kΩ → GPIO pin
#              └→ 2kΩ → GND
#   This brings 5V down to ~3.3V safely.
#
# gpiozero.DistanceSensor docs:
#   DistanceSensor(echo, trigger, max_distance, threshold_distance)
#   .distance  property returns 0.0–1.0 as fraction of max_distance
#   Multiply by max_distance to get metres.
#
# Scaling:
#   raw_fraction = sensor.distance          # 0.0 – 1.0
#   raw_metres   = raw_fraction * MAX_DIST  # 0.0 – MAX_DIST metres
#   Published value is raw_metres, clamped to [MIN_RANGE_M, MAX_RANGE_M].
#   Readings outside valid range are discarded (not published).
#
# Published topics:
#   /ultrasonic/1  through  /ultrasonic/6   std_msgs/Float32  (metres)
#
# Usage:
#   python3 gpio_ultrasonic_node.py
#
# Dependencies:
#   pip install gpiozero
#   sudo apt install python3-rpi.gpio   # backend for gpiozero on Pi
# =============================================================================

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    from gpiozero import DistanceSensor
    from gpiozero.exc import DistanceSensorNoEcho
    GPIO_AVAILABLE = True
except ImportError:
    # Allow the file to be imported on non-Pi machines for syntax checking
    GPIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# GPIO pin assignments
# Edit these to match your physical wiring.
# Uses BCM (Broadcom) pin numbering — this is gpiozero's default.
#
# Sensor index → (echo_bcm_pin, trigger_bcm_pin)
# ---------------------------------------------------------------------------
SENSOR_GPIO_PINS = {
    1: (24, 23),   # echo, trigger
    2: (16, 20),
    3: (12, 21),
    4: (25, 8),
    5: (11, 7),
    6: (9,  10),
}

# HC-SR04 absolute range limits in metres
MIN_RANGE_M = 0.02   # 2 cm  — closer readings are noise
MAX_RANGE_M = 4.00   # 400 cm — rated maximum

# gpiozero max_distance parameter — sets the 1.0 scale ceiling
# Set slightly above MAX_RANGE_M so the sensor never clips at 1.0
GPIOZERO_MAX_DISTANCE = 1   # metres

# How often to read sensors and publish (Hz)
PUBLISH_RATE_HZ = 50.0

# Number of consecutive readings to average per publish cycle.
# Reduces noise from single spurious echo reflections.
# Set to 1 to disable averaging and publish raw readings.
AVERAGE_WINDOW = 10

class GPIOUltrasonicNode(Node):

    def __init__(self):
        super().__init__("gpio_ultrasonic_node")

        if not GPIO_AVAILABLE:
            self.get_logger().fatal(
                "gpiozero is not installed or not available. "
                "Run: pip install gpiozero && sudo apt install python3-rpi.gpio"
            )
            raise RuntimeError("gpiozero unavailable")

        # --- Declare parameters ---
        self.declare_parameter("publish_rate",    float(PUBLISH_RATE_HZ))
        self.declare_parameter("average_window",  AVERAGE_WINDOW)

        self._rate   = self.get_parameter("publish_rate").value
        self._window = int(self.get_parameter("average_window").value)

        # --- Initialise sensors ---
        # gpiozero starts background echo-timing threads automatically.
        # Sensor objects are kept alive for the lifetime of the node.
        self._sensors: dict[int, DistanceSensor] = {}
        self._ultrasonic_publishers: dict[int, rclpy.publisher.Publisher] = {}

        # Rolling window buffer for averaging: idx -> list of recent readings
        self._buffers: dict[int, list[float]] = {idx: [] for idx in SENSOR_GPIO_PINS}

        for idx, (echo_pin, trig_pin) in SENSOR_GPIO_PINS.items():
            try:
                sensor = DistanceSensor(
                    echo=echo_pin,
                    trigger=trig_pin,
                    max_distance=GPIOZERO_MAX_DISTANCE,
                    threshold_distance=0.5,  # not used for distance reading, just events
                )
                self._sensors[idx] = sensor
                self.get_logger().info(
                    f"Sensor {idx} initialised — ECHO: GPIO{echo_pin}, TRIG: GPIO{trig_pin}"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to initialise sensor {idx} "
                    f"(ECHO: GPIO{echo_pin}, TRIG: GPIO{trig_pin}): {e}"
                )
                # Continue — other sensors can still work

            topic = f"/ultrasonic/_{idx}"
            self._ultrasonic_publishers[idx] = self.create_publisher(Float32, topic, 10)
            self.get_logger().info(f"Publishing on {topic}")

        # --- Timer ---
        self.create_timer(1.0 / self._rate, self._tick)

        self.get_logger().info(
            f"GPIO ultrasonic node ready — {len(self._sensors)} sensor(s) active, "
            f"{self._rate}Hz, averaging window={self._window}"
        )

    # -----------------------------------------------------------------------
    # Timer callback
    # -----------------------------------------------------------------------

    def _tick(self):
        """
        Read all sensors, apply averaging, validate range, publish.
        gpiozero updates .distance continuously in its own thread —
        reading the property here is non-blocking.
        """
        for idx, sensor in self._sensors.items():
            try:
                # .distance is a fraction 0.0–1.0 of GPIOZERO_MAX_DISTANCE
                raw_fraction = sensor.distance
            except DistanceSensorNoEcho:
                # No echo received — sensor may be out of range or obstructed
                self.get_logger().warn(
                    f"Sensor {idx}: no echo received — skipping this cycle"
                )
                continue
            except Exception as e:
                self.get_logger().error(f"Sensor {idx} read error: {e}")
                continue

            # Scale fraction to metres
            raw_metres = raw_fraction * GPIOZERO_MAX_DISTANCE

            # Validate against HC-SR04 physical limits
            if raw_metres < MIN_RANGE_M or raw_metres > MAX_RANGE_M:
                # Out of valid range — discard, do not publish
                # This prevents phantom obstacles from noise or out-of-range targets
                self.get_logger().debug(
                    f"Sensor {idx}: reading {raw_metres:.3f}m out of valid range "
                    f"[{MIN_RANGE_M}, {MAX_RANGE_M}] — discarded"
                )
                continue

            # --- Rolling average ---
            buf = self._buffers[idx]
            buf.append(raw_metres)
            if len(buf) > self._window:
                buf.pop(0)  # drop oldest reading

            averaged = sum(buf) / len(buf)

            # --- Publish ---
            msg = Float32()
            msg.data = float(averaged)
            self._ultrasonic_publishers[idx].publish(msg)

            self.get_logger().debug(
                f"Sensor {idx}: raw={raw_metres:.3f}m  avg={averaged:.3f}m"
            )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def destroy_node(self):
        """Close all gpiozero sensor objects cleanly on shutdown."""
        self.get_logger().info("Shutting down — releasing GPIO sensors...")
        for idx, sensor in self._sensors.items():
            try:
                sensor.close()
                self.get_logger().info(f"Sensor {idx} GPIO released.")
            except Exception as e:
                self.get_logger().warn(f"Sensor {idx} close error: {e}")
        super().destroy_node()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = GPIOUltrasonicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()