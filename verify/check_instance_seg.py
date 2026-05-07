import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

TOPIC = "/front_stereo_camera/instance_segmentation"


class InstanceSegChecker(Node):
    def __init__(self):
        super().__init__("instance_seg_checker")
        self.sub = self.create_subscription(Image, TOPIC, self.callback, 10)

    def callback(self, msg: Image):
        print("\n========== Instance Segmentation ==========")
        print(f"stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
        print(f"frame_id: {msg.header.frame_id}")
        print(f"height: {msg.height}")
        print(f"width: {msg.width}")
        print(f"encoding: {msg.encoding}")
        print(f"step: {msg.step}")

        if msg.encoding != "32SC1":
            print(f"[WARN] Expected 32SC1, got {msg.encoding}")

        mask = np.frombuffer(msg.data, dtype=np.int32).reshape(msg.height, msg.width)

        unique_ids, counts = np.unique(mask, return_counts=True)
        pairs = sorted(zip(unique_ids.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)

        print(f"min id: {mask.min()}")
        print(f"max id: {mask.max()}")
        print(f"unique id count: {len(unique_ids)}")

        print("\nTop instance ids by pixel count:")
        for uid, cnt in pairs[:30]:
            print(f"  id={uid:>6}, pixels={cnt}")

        rclpy.shutdown()


def main():
    rclpy.init()
    node = InstanceSegChecker()
    rclpy.spin(node)


if __name__ == "__main__":
    main()