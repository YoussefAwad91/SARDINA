#!/usr/bin/env python3
"""
gpio_bridge_node.py
--------------------
Subscribes to /mech_1 and /mech_2 (std_msgs/Bool).
On True  → sets the corresponding GPIO pin HIGH for PULSE_DURATION_SEC,
            then returns it LOW automatically (one-shot pulse).
On False → no-op (pin already goes LOW after the timer).

GPIO Pin Recommendations for RPi 5
------------------------------------
  Option A (default): GPIO 17 & 27  — safe, widely available
  Option B:           GPIO 22 & 23  — good if 17/27 used by SPI/I2C devices
  Option C:           GPIO 24 & 25  — good if UART (14/15) or SPI (8-11) active
  Option D:           GPIO  5 &  6  — upper header, away from common boot pins
  Option E:           GPIO 16 & 26  — bottom of header, avoids most conflicts

  AVOID: 2,3 (I2C), 14,15 (UART), 8-11 (SPI), 4 (1-Wire), 18 (PWM0), 19 (PWM1)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    import gpiod
    GPIOD_AVAILABLE = True
    GPIOD_V2 = hasattr(gpiod, "request_lines")
except ImportError:
    GPIOD_AVAILABLE = False
    GPIOD_V2 = False


# ──────────────────────────────────────────────
#  CONFIGURE HERE
# ──────────────────────────────────────────────
MECH1_PIN: int        = 17              # BCM pin for /mech_1
MECH2_PIN: int        = 27              # BCM pin for /mech_2
GPIO_CHIP: str        = "/dev/gpiochip0"
PULSE_DURATION_SEC: float = 0.1       # ← set your desired HIGH duration here
# ──────────────────────────────────────────────


class GpioBridgeNode(Node):
    def __init__(self):
        super().__init__("gpio_bridge_node")

        self.declare_parameter("mech1_pin",          MECH1_PIN)
        self.declare_parameter("mech2_pin",          MECH2_PIN)
        self.declare_parameter("gpio_chip",          GPIO_CHIP)
        self.declare_parameter("pulse_duration_sec", PULSE_DURATION_SEC)

        self._pin1     = self.get_parameter("mech1_pin").value
        self._pin2     = self.get_parameter("mech2_pin").value
        self._duration = self.get_parameter("pulse_duration_sec").value
        chip_path      = self.get_parameter("gpio_chip").value

        # Always init to None — prevents AttributeError if GPIO setup fails
        self._gpio_ok = False
        self._chip    = None
        self._line1   = None
        self._line2   = None
        self._lines   = None   # gpiod v2

        # One timer handle per channel; None means pin is not currently pulsing
        self._timer1 = None
        self._timer2 = None

        if not GPIOD_AVAILABLE:
            self.get_logger().error(
                "Python package 'gpiod' not found. "
                "Install: pip install gpiod  (needs libgpiod2 on the host)"
            )
        else:
            self.get_logger().info(f"Detected gpiod {'v2' if GPIOD_V2 else 'v1'} API")
            try:
                if GPIOD_V2:
                    self._init_gpio_v2(chip_path)
                else:
                    self._init_gpio_v1(chip_path)
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to initialise GPIO on '{chip_path}': {exc}\n"
                    "Ensure /dev/gpiochip0 is passed to the container:\n"
                    "  docker run --device /dev/gpiochip0 ...\n"
                    "On RPi 5 the main bank may be /dev/gpiochip4 — try:\n"
                    "  --ros-args -p gpio_chip:=/dev/gpiochip4"
                )

        self.create_subscription(Bool, "/mech_1", self._cb_mech1, 10)
        self.create_subscription(Bool, "/mech_2", self._cb_mech2, 10)

        self.get_logger().info(
            f"gpio_bridge_node ready — pulse duration: {self._duration}s | "
            f"mech_1 → GPIO{self._pin1}, mech_2 → GPIO{self._pin2}"
        )

    # ── GPIO init ─────────────────────────────────────────────────────────

    def _init_gpio_v1(self, chip_path: str) -> None:
        self._chip  = gpiod.Chip(chip_path)
        self._line1 = self._chip.get_line(self._pin1)
        self._line2 = self._chip.get_line(self._pin2)
        cfg = gpiod.LineRequest()
        cfg.consumer     = "gpio_bridge"
        cfg.request_type = gpiod.LineRequest.DIRECTION_OUTPUT
        self._line1.request(cfg, default_val=0)
        self._line2.request(cfg, default_val=0)
        self._gpio_ok = True
        self.get_logger().info(f"GPIO ready (v1) — chip: {chip_path}")

    def _init_gpio_v2(self, chip_path: str) -> None:
        self._lines = gpiod.request_lines(
            chip_path,
            consumer="gpio_bridge",
            config={
                (self._pin1, self._pin2): gpiod.LineSettings(
                    direction=gpiod.line.Direction.OUTPUT,
                    output_value=gpiod.line.Value.INACTIVE,
                )
            },
        )
        self._gpio_ok = True
        self.get_logger().info(f"GPIO ready (v2) — chip: {chip_path}")

    # ── Subscriptions ─────────────────────────────────────────────────────

    def _cb_mech1(self, msg: Bool) -> None:
        if msg.data:
            self._trigger_pulse(self._pin1, "mech_1", timer_attr="_timer1")

    def _cb_mech2(self, msg: Bool) -> None:
        if msg.data:
            self._trigger_pulse(self._pin2, "mech_2", timer_attr="_timer2")

    # ── Pulse logic ───────────────────────────────────────────────────────

    def _trigger_pulse(self, pin: int, name: str, timer_attr: str) -> None:
        """Drive pin HIGH and schedule it to go LOW after _duration seconds.
        If a pulse is already running, the timer is reset (re-triggered)."""
        existing_timer = getattr(self, timer_attr)
        if existing_timer is not None:
            existing_timer.cancel()
            self.get_logger().debug(f"{name}: re-triggering pulse (timer reset)")

        self._write_pin(pin, True)
        self.get_logger().info(f"{name} → GPIO{pin} HIGH for {self._duration}s")

        timer = self.create_timer(
            self._duration,
            lambda: self._on_pulse_done(pin, name, timer_attr),
        )
        setattr(self, timer_attr, timer)

    def _on_pulse_done(self, pin: int, name: str, timer_attr: str) -> None:
        """Called once by the timer — drives pin LOW and cancels the timer."""
        timer = getattr(self, timer_attr)
        if timer is not None:
            timer.cancel()
            setattr(self, timer_attr, None)

        self._write_pin(pin, False)
        self.get_logger().info(f"{name} → GPIO{pin} LOW (pulse done)")

    # ── Low-level GPIO write ───────────────────────────────────────────────

    def _write_pin(self, pin: int, state: bool) -> None:
        if not self._gpio_ok:
            return
        try:
            if GPIOD_V2:
                val = gpiod.line.Value.ACTIVE if state else gpiod.line.Value.INACTIVE
                self._lines.set_value(pin, val)
            else:
                line = self._line1 if pin == self._pin1 else self._line2
                line.set_value(1 if state else 0)
        except Exception as exc:
            self.get_logger().error(f"Failed to set GPIO{pin}: {exc}")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        for timer_attr in ("_timer1", "_timer2"):
            t = getattr(self, timer_attr, None)
            if t:
                t.cancel()
        if self._gpio_ok:
            try:
                if GPIOD_V2 and self._lines:
                    self._lines.set_value(self._pin1, gpiod.line.Value.INACTIVE)
                    self._lines.set_value(self._pin2, gpiod.line.Value.INACTIVE)
                    self._lines.release()
                else:
                    for line in (self._line1, self._line2):
                        if line:
                            line.set_value(0)
                            line.release()
                    if self._chip:
                        self._chip.close()
                self.get_logger().info("GPIO pins set LOW and released.")
            except Exception as exc:
                self.get_logger().warning(f"Cleanup error: {exc}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GpioBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()