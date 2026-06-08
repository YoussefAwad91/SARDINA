#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import time

# ─── Tuning Constants ────────────────────────────────────────────────────────
STEP_SIZE        = 0.10   # meters
OVERSHOOT_MARGIN = 0.02   # meters
MIN_VELOCITY     = 0.2   # m/s
STOP_DEBOUNCE_S  = 0.15   # seconds — must see zero for this long before stopping
# ─────────────────────────────────────────────────────────────────────────────

class AxisState:

    def __init__(self):
        self.current_pos    = 0.0
        self.target         = None
        self.active         = False
        self.direction      = 1.0
        self.zero_since     = None   # timestamp when velocity first went zero

    def reset(self):
        self.target     = None
        self.active     = False
        self.zero_since = None

class VelocityToPositionNode(Node):

    def __init__(self):
        super().__init__('vel_to_dist_node')

        self.x = AxisState()
        self.y = AxisState()

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(Twist,   '/final_cmd', self.cmd_callback,   10)
        self.create_subscription(Float32, '/pos/x',     self.pos_x_callback, 10)
        self.create_subscription(Float32, '/pos/y',     self.pos_y_callback, 10)

        # ── Publishers ─────────────────────────────────────────────────────
        self.pub_x = self.create_publisher(Float32, '/cmd_pos/x', 10)
        self.pub_y = self.create_publisher(Float32, '/cmd_pos/y', 10)

        # ── Debounce check timer (runs every 50 ms) ─────────────────────────
        self.create_timer(0.05, self.debounce_timer_cb)

        self.get_logger().info('vel_to_dist_node started ✔')

    # ── Debounce timer ─────────────────────────────────────────────────────
    def debounce_timer_cb(self):
        """
        Fires every 50 ms.
        If an axis has been at zero velocity for longer than STOP_DEBOUNCE_S,
        commit the STOP.
        """
        now = time.monotonic()
        self._check_debounce(self.x, self.pub_x, now, axis='x')
        self._check_debounce(self.y, self.pub_y, now, axis='y')

    def _check_debounce(self, state: AxisState, pub, now: float, axis: str):
        if state.active and state.zero_since is not None:
            elapsed = now - state.zero_since
            if elapsed >= STOP_DEBOUNCE_S:
                freeze_target = state.current_pos
                state.reset()
                self._publish(pub, freeze_target, axis, label='STOP')

    # ── Position feedback ──────────────────────────────────────────────────

    def pos_x_callback(self, msg: Float32):
        self.x.current_pos = msg.data
        self._check_progress(self.x, self.pub_x, axis='x')

    def pos_y_callback(self, msg: Float32):
        self.y.current_pos = msg.data
        self._check_progress(self.y, self.pub_y, axis='y')

    # ── Velocity command ───────────────────────────────────────────────────

    def cmd_callback(self, msg: Twist):
        self._handle_axis(self.x, self.pub_x, msg.linear.x, axis='x')
        self._handle_axis(self.y, self.pub_y, msg.linear.y, axis='y')

    # ── Per-axis logic ─────────────────────────────────────────────────────

    def _handle_axis(self, state: AxisState, pub, velocity: float, axis: str):

        if abs(velocity) > MIN_VELOCITY:

            direction = 1.0 if velocity > 0.0 else -1.0

            # ── Cancel any pending stop debounce ───────────────────────────
            state.zero_since = None

            if not state.active:
                # ── Fresh start ────────────────────────────────────────────
                state.active    = True
                state.direction = direction
                state.target    = state.current_pos + direction * STEP_SIZE
                self._publish(pub, state.target, axis, label='START')

            elif direction != state.direction:
                # ── Direction flipped ──────────────────────────────────────
                state.direction = direction
                state.target    = state.current_pos + direction * STEP_SIZE
                self._publish(pub, state.target, axis, label='DIR FLIP')

        else:
            # ── Velocity zero — start debounce clock ───────────────────────
            if state.active and state.zero_since is None:
                state.zero_since = time.monotonic()

    # ── Rolling step logic ─────────────────────────────────────────────────

    def _check_progress(self, state: AxisState, pub, axis: str):
        """
        Roll target forward by one STEP (in correct direction) whenever
        the robot is within OVERSHOOT_MARGIN of the current target.
        Uses old_target + direction * STEP to keep chain clean.
        """
        if not state.active or state.target is None:
            return

        distance_remaining = (state.target - state.current_pos) * state.direction

        if distance_remaining <= OVERSHOOT_MARGIN:
            state.target = state.target + state.direction * STEP_SIZE
            self._publish(pub, state.target, axis, label='STEP')

    # ── Publish helper ─────────────────────────────────────────────────────

    def _publish(self, pub, value: float, axis: str, label: str):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)
        self.get_logger().info(
            f'[{axis.upper()}] target={value:+.3f} m  ({label})'
        )

# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VelocityToPositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()