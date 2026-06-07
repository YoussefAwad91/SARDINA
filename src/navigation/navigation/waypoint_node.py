#!/usr/bin/env python3
# =============================================================================
# waypoint_node.py  (holonomic)
# Cycles through a pre-programmed list of waypoints autonomously.
#
# Behaviour per waypoint:
#   1. ALIGN  — rotate in place until heading error < ANGLE_THRESHOLD_DEG
#   2. DRIVE  — drive in X *and* Y (holonomic) until dist < POS_THRESHOLD_M
#   3. STOP   — publish zero velocity, advance to next waypoint, repeat
#
# Because motion is holonomic, the robot can strafe to the target without
# needing to face it.  The ALIGN phase is kept so the robot always arrives
# at the waypoint with the desired heading — disable it by setting
# ENABLE_ALIGN = False.
#
# Subscribes:
#   /pos/x        std_msgs/Float32   metres
#   /pos/y        std_msgs/Float32   metres
#   /pos/theta    std_msgs/Float32   degrees
#
# Publishes:
#   /waypoint_cmd geometry_msgs/Twist
#       linear.x  = forward  velocity (m/s, world-frame decomposed)
#       linear.y  = strafe   velocity (m/s, world-frame decomposed)
#       angular.z = angular  velocity (rad/s, proportional — ALIGN only)
# =============================================================================

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

# ---------------------------------------------------------------------------
# Waypoints — edit freely.
# Each tuple: (x_metres, y_metres, theta_degrees)
# theta is the heading the robot should have WHILE approaching this waypoint.
# ---------------------------------------------------------------------------
WAYPOINTS: list[tuple[float, float, float]] = [
    (1.0,  0.0,   0.0),   # 1 m ahead, facing east
    (1.0,  1.0,   0.0),   # strafe north 1 m
    (0.0,  1.0,   0.0),   # strafe west  1 m
    (0.0,  0.0,   0.0),   # strafe south 1 m back to origin
]

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
# Set to False to skip the ALIGN phase and go straight to DRIVE.
ENABLE_ALIGN: bool = True

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
POS_THRESHOLD_M     = 0.05   # metres  — close enough to count as arrived
ANGLE_THRESHOLD_DEG = 5.0    # degrees — close enough to count as aligned

# ---------------------------------------------------------------------------
# Constant speeds (used when proportional gains are disabled / for safety cap)
# ---------------------------------------------------------------------------
LINEAR_SPEED  = 0.25   # m/s   — used as the constant drive speed
ANGULAR_SPEED = 0.5    # rad/s — used in bang-bang ALIGN fallback

# ---------------------------------------------------------------------------
# Proportional gains
# ---------------------------------------------------------------------------
KP_LINEAR  = 0.4    # (m/s)   per metre   of position error
KP_ANGULAR = 0.03   # (rad/s) per degree  of heading error

MAX_LINEAR_VEL  = 0.5   # m/s
MAX_ANGULAR_VEL = 1.0   # rad/s

# ---------------------------------------------------------------------------
# Control loop rate
# ---------------------------------------------------------------------------
CONTROL_RATE_HZ = 20

# ---------------------------------------------------------------------------
# State machine states
# ---------------------------------------------------------------------------
class State:
    ALIGN = "ALIGN"
    DRIVE = "DRIVE"


class WaypointNode(Node):

    def __init__(self):
        super().__init__("waypoint_node")

        # --- Current pose (updated by subscribers) ---
        self._x     = 0.0
        self._y     = 0.0
        self._theta = 0.0   # degrees

        # --- Waypoint tracking ---
        self._waypoint_index = 0
        self._state          = State.ALIGN if ENABLE_ALIGN else State.DRIVE

        # --- Publisher ---
        self._waypoint_cmd_pub = self.create_publisher(Twist, "/waypoint_cmd", 10)

        # --- Subscribers ---
        self.create_subscription(Float32, "/pos/x",     self._cb_x,     10)
        self.create_subscription(Float32, "/pos/y",     self._cb_y,     10)
        self.create_subscription(Float32, "/pos/theta", self._cb_theta, 10)

        # --- Control loop timer ---
        self.create_timer(1.0 / CONTROL_RATE_HZ, self._control_loop)

        self.get_logger().info(
            f"Holonomic waypoint node ready — {len(WAYPOINTS)} waypoints loaded. "
            f"ALIGN={'ON' if ENABLE_ALIGN else 'OFF'}. "
            f"Starting at waypoint 0: {WAYPOINTS[0]}"
        )

    # -----------------------------------------------------------------------
    # Subscriber callbacks
    # -----------------------------------------------------------------------

    def _cb_x(self, msg: Float32):
        self._x = msg.data

    def _cb_y(self, msg: Float32):
        self._y = msg.data

    def _cb_theta(self, msg: Float32):
        self._theta = msg.data

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _current_waypoint(self) -> tuple[float, float, float]:
        return WAYPOINTS[self._waypoint_index]

    def _advance_waypoint(self):
        """Move to the next waypoint, looping back to 0 at the end."""
        self._waypoint_index = (self._waypoint_index + 1) % len(WAYPOINTS)
        self._state = State.ALIGN if ENABLE_ALIGN else State.DRIVE
        self.get_logger().info(
            f"Advancing to waypoint {self._waypoint_index}: "
            f"{self._current_waypoint()}"
        )

    def _publish_cmd(self, linear_x: float, linear_y: float, angular_z: float):
        """Publish a Twist command on /waypoint_cmd."""
        msg = Twist()
        msg.linear.x  = float(linear_x)
        msg.linear.y  = float(linear_y)
        msg.linear.z  = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular_z)
        self._waypoint_cmd_pub.publish(msg)

    @staticmethod
    def _normalise_angle(angle_deg: float) -> float:
        """Wrap an angle in degrees to (-180, 180]."""
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg <= -180.0:
            angle_deg += 360.0
        return angle_deg

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def _world_to_body(self, vx_world: float, vy_world: float) -> tuple[float, float]:
        """
        Rotate a world-frame velocity vector into the robot body frame.

        Body frame:  +X = forward,  +Y = left
        World frame: +X = east,     +Y = north

        [ vx_body ]   [ cos(θ)  sin(θ) ] [ vx_world ]
        [ vy_body ] = [-sin(θ)  cos(θ) ] [ vy_world ]
        """
        theta_rad = math.radians(self._theta)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        vx_body =  cos_t * vx_world + sin_t * vy_world
        vy_body = -sin_t * vx_world + cos_t * vy_world
        return vx_body, vy_body

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------

    def _control_loop(self):
        wp_x, wp_y, wp_theta = self._current_waypoint()

        # --- World-frame position error ---
        dx = wp_x - self._x
        dy = wp_y - self._y
        dist = math.hypot(dx, dy)

        # --- Heading error (degrees, normalised to (-180, 180]) ---
        heading_error = self._normalise_angle(wp_theta - self._theta)

        # ── ALIGN state ────────────────────────────────────────────────────
        if self._state == State.ALIGN:
            if abs(heading_error) < ANGLE_THRESHOLD_DEG:
                # Aligned — stop rotating, switch to DRIVE
                self._publish_cmd(0.0, 0.0, 0.0)
                self._state = State.DRIVE
                self.get_logger().info(
                    f"Aligned to {wp_theta}° — driving to "
                    f"({wp_x}, {wp_y}), dist={dist:.3f} m"
                )
            else:
                # Bang-bang angular velocity — rotate in place, no translation
                angular_z = ANGULAR_SPEED if heading_error > 0 else -ANGULAR_SPEED
                self._publish_cmd(0.0, 0.0, angular_z)
                self.get_logger().debug(
                    f"ALIGN  heading_err={heading_error:.1f}°  "
                    f"angular_z={angular_z:.3f}"
                )

        # ── DRIVE state (holonomic) ─────────────────────────────────────────
        elif self._state == State.DRIVE:
            if dist < POS_THRESHOLD_M:
                # Arrived — publish stop and advance
                self._publish_cmd(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f"Reached waypoint {self._waypoint_index} "
                    f"({wp_x}, {wp_y}) within {dist:.3f} m"
                )
                self._advance_waypoint()
            else:
                # ------------------------------------------------------------------
                # Holonomic drive:
                #   Each axis is commanded independently at LINEAR_SPEED —
                #   the resultant speed can reach LINEAR_SPEED*√2 on diagonals,
                #   which is intentional (each motor works at full capacity).
                #   A per-axis deadband zeroes out an axis that has already
                #   converged so it doesn't oscillate while the other catches up.
                # ------------------------------------------------------------------
                vx_world = math.copysign(LINEAR_SPEED, dx) if abs(dx) > POS_THRESHOLD_M else 0.0
                vy_world = math.copysign(LINEAR_SPEED, dy) if abs(dy) > POS_THRESHOLD_M else 0.0

                # Transform to body frame
                vx_body, vy_body = self._world_to_body(vx_world, vy_world)

                # Clamp each axis independently
                vx_body = self._clamp(vx_body, MAX_LINEAR_VEL)
                vy_body = self._clamp(vy_body, MAX_LINEAR_VEL)

                self._publish_cmd(vx_body, vy_body, 0.0)
                self.get_logger().debug(
                    f"DRIVE  dist={dist:.3f} m  "
                    f"vx_world={vx_world:.3f}  vy_world={vy_world:.3f}  "
                    f"vx_body={vx_body:.3f}  vy_body={vy_body:.3f}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()