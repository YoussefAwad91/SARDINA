#!/usr/bin/env python3
# =============================================================================
# ultrasonic_fusion_node.py
# ROS2 node that fuses 6 HC-SR04 ultrasonic sensors into a single /obstacle
# topic in the format consumed by the robot dashboard.
#
# Sensor geometry:
#   - 6 sensors mounted in a circle, 25cm from robot center
#   - Center angles: 0, 76, 104, 180, 256, 284 degrees
#   - HC-SR04 beam width: 30 degrees total (±15° from center)
#   - Raw reading = distance from sensor face to obstacle
#   - Published reading = distance from robot CENTER to obstacle
#
# Distance correction per ray angle within the beam cone:
#   Each ray at angle offset α from sensor centerline travels through
#   the sensor's mounting offset before reaching the obstacle.
#   We use the law of cosines to compute true center-to-obstacle distance.
#
# Output:
#   /obstacle  std_msgs/String  JSON: {"points": [{"theta": deg, "distance": m}]}
#
# Usage:
#   ros2 run <your_package> ultrasonic_fusion_node
#   or directly: python3 ultrasonic_fusion_node.py
# =============================================================================

import json
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32

# ---------------------------------------------------------------------------
# Sensor configuration
# ---------------------------------------------------------------------------

# Mounting offset from robot center to sensor face, in metres
SENSOR_OFFSET_M = 0.25

# HC-SR04 half-angle beam width in degrees
BEAM_HALF_ANGLE_DEG = 15.0

OFFSET_DEG = 0

# Center angle (degrees) for each sensor, keyed by sensor index 1-6
SENSOR_ANGLES = {
    1: 0+OFFSET_DEG,
    6: 284+OFFSET_DEG,
    5: 256+OFFSET_DEG,
    4: 180+OFFSET_DEG,
    3: 104+OFFSET_DEG,    
    2: 76+OFFSET_DEG,
    
}

# How often to publish the fused /obstacle message (Hz)
PUBLISH_RATE_HZ = 10

# Maximum valid sensor reading in metres (HC-SR04 max range is 4m)
MAX_RANGE_M = 4.0

# Minimum valid sensor reading in metres (HC-SR04 min range is 2cm)
MIN_RANGE_M = 0.02

# Angular resolution of the output in degrees (1 = one point per degree in cone)
ANGULAR_RESOLUTION_DEG = 1

# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def center_distance(d_raw: float, sensor_offset: float, ray_angle_offset_deg: float) -> float:
    """
    Compute the distance from the robot CENTER to the obstacle for a single
    ray within the sensor's beam cone.

    The sensor face is at distance `sensor_offset` from center along the
    sensor's centerline. The obstacle is at distance `d_raw` from the sensor
    face along the ray direction.

    We model this as a triangle:
      - Side a = sensor_offset  (center → sensor face, along centerline)
      - Side b = d_raw          (sensor face → obstacle, along ray)
      - Angle C = ray_angle_offset_deg (angle between the two sides at sensor face)

    Law of cosines:
      c² = a² + b² - 2ab·cos(C)
    where c is the center-to-obstacle distance we want.

    For rays exactly on the sensor centerline (offset = 0°) this simplifies to:
      c = d_raw + sensor_offset  (collinear, just add the offset)
    """
    if abs(ray_angle_offset_deg) < 1e-6:
        # On centerline — simple addition
        return d_raw + sensor_offset

    angle_rad = math.radians(ray_angle_offset_deg)

    # Law of cosines: the angle at the sensor face between
    # the mounting arm (pointing back to center) and the ray (pointing to obstacle)
    # is (180° - ray_angle_offset) because the arm points AWAY from center
    interior_angle_rad = math.pi - angle_rad

    a = sensor_offset
    b = d_raw
    c_sq = a**2 + b**2 - 2 * a * b * math.cos(interior_angle_rad)

    # Clamp to avoid sqrt of negative due to floating point
    return math.sqrt(max(c_sq, 0.0))

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class UltrasonicFusionNode(Node):

    def __init__(self):
        super().__init__("ultrasonic_fusion_node")
        self.get_logger().info("Ultrasonic fusion node starting...")

        # Latest raw reading from each sensor (metres), None = no data yet
        self._readings: dict[int, float | None] = {i: None for i in SENSOR_ANGLES}

        # Subscribe to each sensor topic
        for sensor_idx in SENSOR_ANGLES:
            topic = f"/ultrasonic/_{sensor_idx}"
            self.create_subscription(
                Float32,
                topic,
                # Capture sensor_idx in the lambda with a default argument
                lambda msg, idx=sensor_idx: self._on_sensor(idx, msg),
                10,  # QoS depth
            )
            self.get_logger().info(f"Subscribed to {topic}")

        # Publisher for fused obstacle map
        self._obstacle_pub = self.create_publisher(String, "/obstacle", 10)

        # Timer drives publishing at fixed rate
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_fused)

        self.get_logger().info(
            f"Fusion node ready. Publishing /obstacle at {PUBLISH_RATE_HZ}Hz"
        )

    # -----------------------------------------------------------------------
    # Sensor callbacks
    # -----------------------------------------------------------------------

    def _on_sensor(self, sensor_idx: int, msg: Float32):
        """
        Store the latest raw reading from a sensor.
        Validates range — out-of-range readings are stored as None
        so they do not contribute phantom obstacles.
        """
        raw = msg.data

        if raw < MIN_RANGE_M or raw > MAX_RANGE_M:
            # Reading out of valid HC-SR04 range — discard
            self._readings[sensor_idx] = None
            return

        self._readings[sensor_idx] = raw

    # -----------------------------------------------------------------------
    # Fusion and publishing
    # -----------------------------------------------------------------------

    def _publish_fused(self):
        """
        Called at PUBLISH_RATE_HZ.
        Fuses all available sensor readings into a list of (theta, distance)
        points and publishes as a JSON string on /obstacle.
        """
        points = []

        for sensor_idx, center_angle_deg in SENSOR_ANGLES.items():
            raw = self._readings[sensor_idx]

            if raw is None:
                # No valid reading from this sensor — skip it entirely
                # Do not publish phantom points
                continue

            # Expand the single raw reading across the full beam cone
            # at ANGULAR_RESOLUTION_DEG steps
            cone_points = self._expand_beam(
                raw,
                center_angle_deg,
                BEAM_HALF_ANGLE_DEG,
                ANGULAR_RESOLUTION_DEG,
            )
            points.extend(cone_points)

        # If multiple sensors cover overlapping angles (sensors 2&3 at 76/104
        # overlap slightly at ~89-91°), keep the CLOSER reading per degree
        # since closer = more urgent obstacle
        points = self._resolve_overlaps(points)

        # Publish even if points is empty — dashboard needs to know the coast is clear
        payload = json.dumps({"points": points})
        msg = String()
        msg.data = payload
        self._obstacle_pub.publish(msg)

    def _expand_beam(
        self,
        d_raw: float,
        center_angle_deg: float,
        half_angle_deg: float,
        resolution_deg: int,
    ) -> list[dict]:
        """
        Expand a single sensor reading into multiple (theta, distance) points
        spanning the full beam cone.

        For each degree step within the cone:
          1. Compute the ray's angle offset from sensor centerline
          2. Use law of cosines to find true center-to-obstacle distance
          3. Normalise the output angle to 0-359°

        Returns a list of {"theta": float, "distance": float} dicts.
        """
        points = []

        start = -int(half_angle_deg)
        end   =  int(half_angle_deg) + 1  # inclusive

        for offset_deg in range(start, end, resolution_deg):
            # Absolute world angle of this ray
            world_angle_deg = (center_angle_deg + offset_deg) % 360

            # True distance from robot center to obstacle along this ray
            d_center = center_distance(d_raw, SENSOR_OFFSET_M, offset_deg)

            points.append({
                "theta":    round(float(world_angle_deg), 2),
                "distance": round(float(d_center), 4),
            })

        return points

    def _resolve_overlaps(self, points: list[dict]) -> list[dict]:
        """
        When two sensors produce points at the same angle, keep the closer one.
        Sensors 2 & 3 (76° and 104°) overlap at ~89-91°.
        Sensors 5 & 6 (256° and 284°) overlap at ~269-271°.

        Groups all points by theta, picks minimum distance per theta.
        Returns a flat sorted list.
        """
        best: dict[float, float] = {}  # theta -> minimum distance

        for p in points:
            theta = p["theta"]
            dist  = p["distance"]
            if theta not in best or dist < best[theta]:
                best[theta] = dist

        # Return sorted by angle for clean output
        return [
            {"theta": theta, "distance": dist}
            for theta, dist in sorted(best.items())
        ]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()