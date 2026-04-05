import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher(Node):

    def __init__(self):
        super().__init__('isaac_camera_node')

        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)

        self.bridge = CvBridge()

    def publish(self, rgb, depth):
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
