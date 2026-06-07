#!/usr/bin/env python3
# =============================================================================
# arbiter_node.py
# Central command arbiter for the robot.
#
# Subscribes:
#   /waypoint_cmd            geometry_msgs/Twist  — autonomous motion
#   /cmd_manual              geometry_msgs/Twist  — manual motion
#   /arbiter/manual_override std_msgs/Bool        — False=waypoint, True=manual
#   /obstacle                std_msgs/String      — JSON polar obstacle map
#   /vision/confidence       std_msgs/Float32     — 0.0–1.0
#   /vision/prediction       std_msgs/Int32       — 0–5
#
# Publishes:
#   /final_cmd               geometry_msgs/Twist  — final motion command
#   /mech_1                 std_msgs/Bool        — one-cycle pulse
#   /mech_2                 std_msgs/Bool        — one-cycle pulse
#   /vision/save             std_msgs/Bool        — one-cycle pulse
#
# State machine:
#   NORMAL    → component-wise collision filter, pass clear components
#   DYNAMIC   → obstacle appeared, wait OBSTACLE_TIMEOUT_S to see if transient
#   STATIC    → obstacle confirmed static, check vision confidence
#   AVOIDANCE → low confidence OR retries exhausted, self-generate escape cmd
#
# Prediction mapping:
#   0, 4, 5  → reserved / background class — no action → fall to AVOIDANCE
#   1        → pulse /mech_1
#   2        → pulse /mech_2
#   3        → pulse /vision/save
#
# Vision retry logic:
#   Each fired action resets to DYNAMIC to check if obstacle cleared.
#   If obstacle persists after MAX_VISION_RETRIES firings → AVOIDANCE.
#   If prediction is reserved/background and stable → AVOIDANCE.
#
# Avoidance command priority (intent-aware):
#   Case A — X blocked, no Y intent  : try Y strafe → opposite X → rotate
#   Case B — Y blocked, no X intent  : try X fwd/bwd → opposite Y → rotate
#   Case C — both blocked / diagonal : sort remaining cardinals by clearance
#   Case D — rotation only           : sort remaining cardinals by clearance
# =============================================================================

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String, Int32
from geometry_msgs.msg import Twist

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

CONTROL_RATE_HZ            = 20

DANGER_DISTANCE_M          = 0.5    # metres  — closer than this = blocked
COLLISION_CHECK_ANGLE_DEG  = 20.0   # ± degrees around a cardinal direction

OBSTACLE_TIMEOUT_S         = 3.0    # seconds before obstacle declared static

CONFIDENCE_THRESHOLD       = 0.7    # below → avoidance, at/above → mech action
PREDICTION_STABLE_TIME_S   = 1.5    # prediction must hold this long to act
MAX_VISION_RETRIES         = 3      # firings with no clearance before AVOIDANCE

AVOIDANCE_STRAFE_VEL       = 0.2    # m/s   — linear escape velocity
AVOIDANCE_ROTATE_VEL       = 0.4    # rad/s — rotation escape velocity

VEL_DEADZONE               = 1e-3   # components smaller than this = zero

# ---------------------------------------------------------------------------
# State labels
# ---------------------------------------------------------------------------

class State:
    NORMAL    = "NORMAL"
    DYNAMIC   = "DYNAMIC"
    STATIC    = "STATIC"
    AVOIDANCE = "AVOIDANCE"

# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    """Signed shortest difference a − b in degrees, result in (−180, 180]."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d

def _sector_blocked(points: list[dict],
                    direction_deg: float,
                    half_window_deg: float,
                    danger_dist_m: float) -> bool:
    """
    Return True if ANY obstacle point is within ±half_window_deg of
    direction_deg AND closer than danger_dist_m.
    """
    for p in points:
        if (abs(_angle_diff(p["theta"], direction_deg)) <= half_window_deg
                and p["distance"] < danger_dist_m):
            return True
    return False

def _sector_min_distance(points: list[dict],
                          direction_deg: float,
                          half_window_deg: float) -> float:
    """
    Minimum obstacle distance within ±half_window_deg of direction_deg.
    Returns float('inf') if no points fall in that sector.
    """
    min_d = float("inf")
    for p in points:
        if abs(_angle_diff(p["theta"], direction_deg)) <= half_window_deg:
            min_d = min(min_d, p["distance"])
    return min_d

def _find_clearest_rotation_sign(points: list[dict],
                                  danger_dist_m: float,
                                  half_window_deg: float) -> float:
    """
    Choose rotation direction when all linear escapes are blocked.
    Returns +1.0 (CCW) or -1.0 (CW) based on which lateral side has
    more clearance.
    """
    left_blocked  = _sector_blocked(points,  90.0, half_window_deg, danger_dist_m)
    right_blocked = _sector_blocked(points, 270.0, half_window_deg, danger_dist_m)

    if not left_blocked and right_blocked:
        return  1.0   # CCW — left is clearly freer
    if not right_blocked and left_blocked:
        return -1.0   # CW  — right is clearly freer

    # Both same state — pick the side with the greater minimum distance
    left_dist  = _sector_min_distance(points,  90.0, half_window_deg)
    right_dist = _sector_min_distance(points, 270.0, half_window_deg)
    return 1.0 if left_dist >= right_dist else -1.0

# ---------------------------------------------------------------------------
# Intent-aware avoidance command builder
# ---------------------------------------------------------------------------

def build_avoidance_cmd(intended: Twist,
                         points: list[dict],
                         danger_dist_m: float,
                         half_window_deg: float,
                         strafe_vel: float,
                         rotate_vel: float) -> Twist:
    """
    Build an escape Twist that is aware of what the robot was trying to do.

    Case A — intended had X motion (forward/backward blocked):
        Priority: perpendicular strafe (Y axis) → opposite X → rotate

    Case B — intended had Y motion (strafe blocked):
        Priority: forward/backward (X axis) → opposite Y → rotate

    Case C — intended had both X and Y (diagonal, both blocked):
        Priority: sort all remaining cardinals by clearance distance

    Case D — rotation only or unknown:
        Priority: sort all remaining cardinals by clearance distance

    The direction already confirmed blocked is skipped immediately.
    Returns a Twist with exactly one non-zero component.
    """
    hw = half_window_deg
    d  = danger_dist_m

    has_x = abs(intended.linear.x) > VEL_DEADZONE
    has_y = abs(intended.linear.y) > VEL_DEADZONE

    # Directions that were the intended (already blocked) ones
    intended_x_dir = (0.0  if intended.linear.x >= 0 else 180.0) if has_x else None
    intended_y_dir = (90.0 if intended.linear.y >= 0 else 270.0) if has_y else None

    # Full set of cardinal escape options: direction → (axis, velocity)
    all_dirs = {
        0.0:   ("x",  strafe_vel),
        180.0: ("x", -strafe_vel),
        90.0:  ("y",  strafe_vel),
        270.0: ("y", -strafe_vel),
    }

    # Remove already-blocked intended directions
    blocked_dirs = set()
    if intended_x_dir is not None:
        blocked_dirs.add(intended_x_dir)
    if intended_y_dir is not None:
        blocked_dirs.add(intended_y_dir)

    remaining = {k: v for k, v in all_dirs.items() if k not in blocked_dirs}

    def _as_list(d_map):
        return [(deg, ax, vel) for deg, (ax, vel) in d_map.items()]

    if has_x and not has_y:
        # Case A: X was blocked → try Y-axis escapes first, then opposite X
        perp   = [(deg, ax, vel) for deg, (ax, vel) in remaining.items()
                  if ax == "y"]
        parall = [(deg, ax, vel) for deg, (ax, vel) in remaining.items()
                  if ax == "x"]
        candidates = perp + parall

    elif has_y and not has_x:
        # Case B: Y was blocked → try X-axis escapes first, then opposite Y
        x_dirs = [(deg, ax, vel) for deg, (ax, vel) in remaining.items()
                  if ax == "x"]
        y_dirs = [(deg, ax, vel) for deg, (ax, vel) in remaining.items()
                  if ax == "y"]
        candidates = x_dirs + y_dirs

    else:
        # Case C/D: sort by clearance distance (most clear first)
        candidates = sorted(
            _as_list(remaining),
            key=lambda e: _sector_min_distance(points, e[0], hw),
            reverse=True,
        )

    # Walk candidates and return the first unblocked direction
    for direction_deg, axis, velocity in candidates:
        if not _sector_blocked(points, direction_deg, hw, d):
            cmd = Twist()
            if axis == "x":
                cmd.linear.x = velocity
            else:
                cmd.linear.y = velocity
            return cmd

    # All linear directions blocked — rotate toward the clearer side
    cmd = Twist()
    cmd.angular.z = _find_clearest_rotation_sign(points, d, hw) * rotate_vel
    return cmd

# ---------------------------------------------------------------------------
# Arbiter node
# ---------------------------------------------------------------------------

class ArbiterNode(Node):

    def __init__(self):
        super().__init__("arbiter_node")

        # ── Incoming data ────────────────────────────────────────────────────
        self._manual_override:  bool       = False
        self._waypoint_cmd:     Twist      = Twist()
        self._manual_cmd:       Twist      = Twist()
        self._obstacle_points:  list[dict] = []
        self._confidence:       float      = 0.0
        self._prediction:       int        = 0

        # ── State machine ────────────────────────────────────────────────────
        self._state: str = State.NORMAL

        self._obstacle_first_seen:     float | None = None
        self._last_prediction_value:   int          = -1
        self._prediction_stable_since: float | None = None
        self._vision_retry_count:      int          = 0

        # One-cycle output pulses
        self._pulse_mech1:       bool = False
        self._pulse_mech2:       bool = False
        self._pulse_vision_save: bool = False

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            Twist,   "/waypoint_cmd",            self._cb_waypoint,   10)
        self.create_subscription(
            Twist,   "/cmd_manual",              self._cb_manual,     10)
        self.create_subscription(
            Bool,    "/arbiter/manual_override", self._cb_override,   10)
        self.create_subscription(
            String,  "/obstacle",                self._cb_obstacle,   10)
        self.create_subscription(
            Float32, "/vision/confidence",       self._cb_confidence, 10)
        self.create_subscription(
            Int32,   "/vision/prediction",       self._cb_prediction, 10)

        # ── Publishers ───────────────────────────────────────────────────────
        self._pub_final    = self.create_publisher(Twist, "/final_cmd",   10)
        self._pub_mech1    = self.create_publisher(Bool,  "/mech_1",     10)
        self._pub_mech2    = self.create_publisher(Bool,  "/mech_2",     10)
        self._pub_vis_save = self.create_publisher(Bool,  "/vision/save", 10)

        # ── Control loop ─────────────────────────────────────────────────────
        self.create_timer(1.0 / CONTROL_RATE_HZ, self._control_loop)

        self.get_logger().info(
            f"Arbiter ready | rate={CONTROL_RATE_HZ}Hz "
            f"danger={DANGER_DISTANCE_M}m "
            f"timeout={OBSTACLE_TIMEOUT_S}s "
            f"conf_thresh={CONFIDENCE_THRESHOLD} "
            f"max_retries={MAX_VISION_RETRIES}"
        )

    # =========================================================================
    # Subscriber callbacks
    # =========================================================================

    def _cb_waypoint(self, msg: Twist):
        self._waypoint_cmd = msg

    def _cb_manual(self, msg: Twist):
        self._manual_cmd = msg

    def _cb_override(self, msg: Bool):
        prev = self._manual_override
        self._manual_override = msg.data
        if prev != msg.data:
            self.get_logger().info(
                f"Override → {'MANUAL' if msg.data else 'WAYPOINT'}"
            )

    def _cb_obstacle(self, msg: String):
        try:
            self._obstacle_points = json.loads(msg.data).get("points", [])
        except (json.JSONDecodeError, AttributeError):
            self.get_logger().warn("Malformed /obstacle message — ignoring.")
            self._obstacle_points = []

    def _cb_confidence(self, msg: Float32):
        self._confidence = float(msg.data)

    def _cb_prediction(self, msg: Int32):
        value = int(msg.data)
        now   = self._now()
        if value != self._last_prediction_value:
            self.get_logger().info(
                f"Prediction changed {self._last_prediction_value} → {value} "
                f"— stability timer reset."
            )
            self._last_prediction_value   = value
            self._prediction_stable_since = now
        self._prediction = value

    # =========================================================================
    # Control loop
    # =========================================================================

    def _control_loop(self):
        self._publish_pulses()

        active_cmd = (self._manual_cmd
                      if self._manual_override
                      else self._waypoint_cmd)

        if self._state == State.NORMAL:
            self._run_normal(active_cmd)
        elif self._state == State.DYNAMIC:
            self._run_dynamic(active_cmd)
        elif self._state == State.STATIC:
            self._run_static(active_cmd)
        elif self._state == State.AVOIDANCE:
            self._run_avoidance(active_cmd)

    # =========================================================================
    # State: NORMAL
    # =========================================================================

    def _run_normal(self, cmd: Twist):
        filtered, all_blocked = self._filter_command(cmd)

        if all_blocked:
            self._vision_retry_count  = 0
            self._obstacle_first_seen = self._now()
            self._transition(State.DYNAMIC)
            self._publish_zero()
            return

        self._publish_cmd(filtered)

    # =========================================================================
    # State: DYNAMIC
    # =========================================================================

    def _run_dynamic(self, cmd: Twist):
        _, all_blocked = self._filter_command(cmd)

        if not all_blocked:
            self.get_logger().info(
                "Obstacle cleared during DYNAMIC window — resuming NORMAL."
            )
            self._obstacle_first_seen = None
            self._vision_retry_count  = 0
            self._transition(State.NORMAL)
            filtered, _ = self._filter_command(cmd)
            self._publish_cmd(filtered)
            return

        elapsed = self._now() - self._obstacle_first_seen
        if elapsed >= OBSTACLE_TIMEOUT_S:
            if self._vision_retry_count >= MAX_VISION_RETRIES:
                self.get_logger().info(
                    f"Vision action fired {self._vision_retry_count} times with "
                    f"no clearance — giving up, entering AVOIDANCE."
                )
                self._vision_retry_count = 0
                self._transition(State.AVOIDANCE)
            else:
                self.get_logger().info(
                    f"Obstacle persisted {elapsed:.1f}s — confirmed STATIC "
                    f"(retry {self._vision_retry_count}/{MAX_VISION_RETRIES})."
                )
                self._prediction_stable_since = None
                self._last_prediction_value   = -1
                self._transition(State.STATIC)

        self._publish_zero()

    # =========================================================================
    # State: STATIC
    # =========================================================================

    def _run_static(self, cmd: Twist):
        """
        Obstacle is confirmed static.

        • Obstacle spontaneously clears          → NORMAL
        • Confidence ≥ threshold
            - Prediction is actionable (1/2/3)
              and stable ≥ PREDICTION_STABLE_TIME_S → fire pulse → DYNAMIC
            - Prediction is reserved (0/4/5)
              and stable ≥ PREDICTION_STABLE_TIME_S → AVOIDANCE
            - Not yet stable                         → wait
        • Confidence < threshold                 → AVOIDANCE
        """
        _, all_blocked = self._filter_command(cmd)

        if not all_blocked:
            self.get_logger().info(
                "Obstacle cleared during STATIC phase — resuming NORMAL."
            )
            self._vision_retry_count = 0
            self._transition(State.NORMAL)
            filtered, _ = self._filter_command(cmd)
            self._publish_cmd(filtered)
            return

        if self._confidence >= CONFIDENCE_THRESHOLD:
            fired = self._try_fire_vision_action()

            if fired:
                self._vision_retry_count += 1
                self.get_logger().info(
                    f"Vision action fired (attempt {self._vision_retry_count}"
                    f"/{MAX_VISION_RETRIES}) — resetting to DYNAMIC to check clearance."
                )
                self._obstacle_first_seen     = self._now()
                self._prediction_stable_since = None
                self._last_prediction_value   = -1
                self._transition(State.DYNAMIC)

            else:
                # Not fired — check if prediction is stable but reserved
                if (self._prediction_stable_since is not None
                        and (self._now() - self._prediction_stable_since)
                        >= PREDICTION_STABLE_TIME_S):
                    self.get_logger().info(
                        f"Prediction {self._prediction} is reserved/background "
                        f"and stable — falling through to AVOIDANCE."
                    )
                    self._vision_retry_count = 0
                    self._transition(State.AVOIDANCE)

        else:
            self.get_logger().info(
                f"Confidence {self._confidence:.2f} below threshold "
                f"{CONFIDENCE_THRESHOLD} — entering AVOIDANCE."
            )
            self._vision_retry_count = 0
            self._transition(State.AVOIDANCE)

        self._publish_zero()

    def _try_fire_vision_action(self) -> bool:
        """
        Check whether the current prediction has been stable long enough
        and is an actionable class.

        Returns True  ONLY if an actual pulse was sent (pred 1, 2, or 3).
        Returns False for reserved/background predictions (0, 4, 5) or if
        the prediction has not yet been stable for PREDICTION_STABLE_TIME_S.
        """
        now = self._now()

        # Initialise stability tracking on first call after entering STATIC
        if self._prediction_stable_since is None:
            self._prediction_stable_since = now
            self._last_prediction_value   = self._prediction
            return False

        stable_for = now - self._prediction_stable_since

        if stable_for < PREDICTION_STABLE_TIME_S:
            self.get_logger().debug(
                f"Prediction {self._prediction} stable for "
                f"{stable_for:.2f}/{PREDICTION_STABLE_TIME_S}s — waiting."
            )
            return False

        pred = self._prediction
        self.get_logger().info(
            f"Prediction {pred} stable {stable_for:.2f}s "
            f"(conf={self._confidence:.2f}) — evaluating."
        )

        if pred == 1:
            self._pulse_mech1 = True
            self.get_logger().info("Firing /mech_1")
            return True
        elif pred == 2:
            self._pulse_mech2 = True
            self.get_logger().info("Firing /mech_2")
            return True
        elif pred == 3:
            self._pulse_vision_save = True
            self.get_logger().info("Firing /vision/save")
            return True
        elif pred in (0, 4, 5):
            # Reserved / background class — no pulse, caller handles fallthrough
            self.get_logger().info(
                f"Prediction {pred} is a reserved/background class — no action."
            )
            return False
        else:
            self.get_logger().warn(f"Unknown prediction value: {pred}")
            return False

    # =========================================================================
    # State: AVOIDANCE
    # =========================================================================

    def _run_avoidance(self, cmd: Twist):
        """
        Exit conditions (priority order):
          1. Confidence recovered → STATIC (re-evaluate vision)
          2. Any component of intended command now clear → NORMAL

        Avoidance command is intent-aware — picks the best escape direction
        based on which axis was originally blocked.
        """
        # Exit 1 — confidence recovered
        if self._confidence >= CONFIDENCE_THRESHOLD:
            self.get_logger().info(
                "Confidence recovered during AVOIDANCE — returning to STATIC."
            )
            self._prediction_stable_since = None
            self._last_prediction_value   = -1
            self._transition(State.STATIC)
            self._publish_zero()
            return

        # Exit 2 — intended path partially clear
        filtered, all_blocked = self._filter_command(cmd)
        if not all_blocked:
            self.get_logger().info(
                "Intended path partially clear — exiting AVOIDANCE to NORMAL."
            )
            self._transition(State.NORMAL)
            self._publish_cmd(filtered)
            return

        # Still fully blocked — generate intent-aware avoidance command
        avoidance_cmd = build_avoidance_cmd(
            intended        = cmd,
            points          = self._obstacle_points,
            danger_dist_m   = DANGER_DISTANCE_M,
            half_window_deg = COLLISION_CHECK_ANGLE_DEG,
            strafe_vel      = AVOIDANCE_STRAFE_VEL,
            rotate_vel      = AVOIDANCE_ROTATE_VEL,
        )

        self.get_logger().debug(
            f"AVOIDANCE cmd: lx={avoidance_cmd.linear.x:.2f} "
            f"ly={avoidance_cmd.linear.y:.2f} "
            f"az={avoidance_cmd.angular.z:.2f}"
        )
        self._publish_cmd(avoidance_cmd)

    # =========================================================================
    # Component-wise collision filter
    # =========================================================================

    def _filter_command(self, cmd: Twist) -> tuple[Twist, bool]:
        """
        Check each motion component independently against the obstacle map.

          linear.x > 0  → forward   0°
          linear.x < 0  → backward  180°
          linear.y > 0  → left      90°
          linear.y < 0  → right     270°
          angular.z     → rotation  (blocked only if BOTH lateral sides blocked)

        Returns (filtered_twist, all_blocked).
        A zero command returns (zero_twist, False) — idle never triggers
        obstacle mode.
        """
        pts = self._obstacle_points
        hw  = COLLISION_CHECK_ANGLE_DEG
        d   = DANGER_DISTANCE_M

        out = Twist()

        has_x     = abs(cmd.linear.x)  > VEL_DEADZONE
        has_y     = abs(cmd.linear.y)  > VEL_DEADZONE
        has_omega = abs(cmd.angular.z) > VEL_DEADZONE

        if not has_x and not has_y and not has_omega:
            return out, False

        requested = 0
        blocked   = 0

        if has_x:
            requested += 1
            direction  = 0.0 if cmd.linear.x > 0 else 180.0
            if _sector_blocked(pts, direction, hw, d):
                self.get_logger().debug(
                    f"linear.x ({'fwd' if cmd.linear.x > 0 else 'bwd'}) BLOCKED"
                )
                blocked += 1
            else:
                out.linear.x = cmd.linear.x

        if has_y:
            requested += 1
            direction  = 90.0 if cmd.linear.y > 0 else 270.0
            if _sector_blocked(pts, direction, hw, d):
                self.get_logger().debug(
                    f"linear.y ({'left' if cmd.linear.y > 0 else 'right'}) BLOCKED"
                )
                blocked += 1
            else:
                out.linear.y = cmd.linear.y

        if has_omega:
            requested += 1
            left_blocked  = _sector_blocked(pts,  90.0, hw, d)
            right_blocked = _sector_blocked(pts, 270.0, hw, d)
            if left_blocked and right_blocked:
                self.get_logger().debug("angular.z BLOCKED (surrounded)")
                blocked += 1
            else:
                out.angular.z = cmd.angular.z

        return out, (blocked == requested)

    # =========================================================================
    # Publishing helpers
    # =========================================================================

    def _publish_cmd(self, cmd: Twist):
        self._pub_final.publish(cmd)

    def _publish_zero(self):
        self._pub_final.publish(Twist())

    def _publish_pulses(self):
        def _b(val: bool) -> Bool:
            m = Bool(); m.data = val; return m

        self._pub_mech1.publish(_b(self._pulse_mech1))
        self._pub_mech2.publish(_b(self._pulse_mech2))
        self._pub_vis_save.publish(_b(self._pulse_vision_save))

        self._pulse_mech1       = False
        self._pulse_mech2       = False
        self._pulse_vision_save = False

    # =========================================================================
    # Utility
    # =========================================================================

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _transition(self, new_state: str):
        if new_state != self._state:
            self.get_logger().info(f"State: {self._state} → {new_state}")
            self._state = new_state

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()