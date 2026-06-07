import time
import lgpio

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32


TRIG = 23
ECHO = 24


class UltrasonicNode(Node):

    def __init__(self):

        super().__init__('ultrasonic_node')

        self.publisher_ = self.create_publisher(
            Float32,
            'ultrasonic_distance',
            10
        )

        self.last_distance = 0.0

        self.h = lgpio.gpiochip_open(0)

        lgpio.gpio_claim_output(self.h, TRIG)
        lgpio.gpio_claim_input(self.h, ECHO)

        lgpio.gpio_write(self.h, TRIG, 0)

        time.sleep(2)

        self.timer = self.create_timer(
            0.2,
            self.publish_distance
        )

        self.get_logger().info(
            'Ultrasonic node started'
        )

    def get_distance(self):

        try:

            lgpio.gpio_write(self.h, TRIG, 0)
            time.sleep(0.000002)

            lgpio.gpio_write(self.h, TRIG, 1)
            time.sleep(0.00001)
            lgpio.gpio_write(self.h, TRIG, 0)

            start_time = time.time()

            while lgpio.gpio_read(self.h, ECHO) == 0:

                if time.time() - start_time > 0.05:
                    return None

            pulse_start = time.time()

            while lgpio.gpio_read(self.h, ECHO) == 1:

                if time.time() - pulse_start > 0.05:
                    return None

            pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start

            distance = pulse_duration * 17150

            return round(distance, 2)

        except Exception as e:

            self.get_logger().warning(
                f'Sensor error: {e}'
            )

            return None

    def publish_distance(self):

        distance = self.get_distance()

        if distance is not None:
            self.last_distance = distance

        msg = Float32()

        msg.data = float(self.last_distance)

        self.publisher_.publish(msg)

        self.get_logger().info(
            f'Distance: {self.last_distance:.2f} cm'
        )

    def destroy_node(self):

        try:
            lgpio.gpiochip_close(self.h)
        except Exception:
            pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = UltrasonicNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()