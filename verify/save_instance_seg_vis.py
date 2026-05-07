import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2

TOPIC = "/front_stereo_camera/instance_segmentation"


class InstanceSegSaver(Node):
    def __init__(self):
        super().__init__("instance_seg_saver")
        self.sub = self.create_subscription(Image, TOPIC, self.callback, 10)

    def callback(self, msg: Image):
        mask = np.frombuffer(msg.data, dtype=np.int32).reshape(msg.height, msg.width)

        np.save("instance_seg_raw.npy", mask)

        unique_ids = np.unique(mask)
        print(f"unique ids: {unique_ids[:50]}")
        print(f"unique count: {len(unique_ids)}")

        rng = np.random.default_rng(12345)
        color_map = {}

        for uid in unique_ids:
            uid = int(uid)
            if uid == 0:
                color_map[uid] = np.array([0, 0, 0], dtype=np.uint8)
            else:
                color_map[uid] = rng.integers(30, 255, size=3, dtype=np.uint8)

        vis = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)

        for uid in unique_ids:
            uid = int(uid)
            vis[mask == uid] = color_map[uid]

        cv2.imwrite("instance_seg_vis.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        print("Saved:")
        print("  instance_seg_raw.npy")
        print("  instance_seg_vis.png")

        rclpy.shutdown()


def main():
    rclpy.init()
    node = InstanceSegSaver()
    rclpy.spin(node)


if __name__ == "__main__":
    main()