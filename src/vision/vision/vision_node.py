#!/usr/bin/env python3

import cv2
import rclpy

from rclpy.node import Node
from std_msgs.msg import Int32, Float32
from sensor_msgs.msg import CompressedImage
import glob

from .inference import FingerPredictor
import subprocess

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # Publishers
        self.prediction_pub = self.create_publisher(
            Int32,
            'vision/prediction',
            10
        )

        self.confidence_pub = self.create_publisher(
            Float32,
            'vision/confidence',
            10
        )

        self.frame_pub = self.create_publisher(
            CompressedImage,
            'vision/frame',
            10
        )

        # JPEG quality — lower = smaller payload = less latency
        self.declare_parameter('jpeg_quality', 80)

        camera_index = 0
        self.declare_parameter('brightness', 100)
        brightness = self.get_parameter('brightness').value

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{camera_index}", f"--set-ctrl=brightness={brightness}"])


        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            raise RuntimeError("Camera not available")

        # Predictor
        self.predictor = FingerPredictor()

        # Timer loop (~30 FPS)
        self.timer = self.create_timer(
            1.0 / 30.0,
            self.process_frame
        )

        self.get_logger().info("Vision node started")

    def process_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning("Failed to read frame")
            return

        # Mirror image
        frame = cv2.flip(frame, 1)

        # Encode to JPEG and publish
        quality = self.get_parameter('jpeg_quality').value
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            msg = CompressedImage()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            msg.format          = 'jpeg'
            msg.data            = buf.tobytes()
            self.frame_pub.publish(msg)

        # Run inference
        count, confidence = self.predictor.predict(frame)

        # Publish prediction
        pred_msg      = Int32()
        pred_msg.data = int(count)
        self.prediction_pub.publish(pred_msg)

        # Publish confidence
        conf_msg      = Float32()
        conf_msg.data = float(confidence)
        self.confidence_pub.publish(conf_msg)

    def destroy_node(self):

        if self.cap.isOpened():
            self.cap.release()

        super().destroy_node()

def main(args=None):

    rclpy.init(args=args)
    node = VisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()