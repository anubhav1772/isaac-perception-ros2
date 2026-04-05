import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
from object_detection.yolo_detector import YOLOv8Detector
import struct
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()
        self.detector = YOLOv8Detector("yolov8n.pt")  # use your trained model

        self.rgb_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/rgb/camera_info', self.cam_info_callback, 10)

        self.pc_pub = self.create_publisher(PointCloud2, '/detected_objects', 10)

        self.latest_rgb = None
        self.latest_depth = None
        self.cam_info = None

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def cam_info_callback(self, msg):
        self.cam_info = msg

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        self.try_process()

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.try_process()

    def try_process(self):
        if self.latest_rgb is None or self.latest_depth is None or self.cam_info is None:
            return

        rgb = self.latest_rgb
        depth = self.latest_depth
        fx = self.cam_info.k[0]
        fy = self.cam_info.k[4]
        cx = self.cam_info.k[2]
        cy = self.cam_info.k[5]

        detections = self.detector.detect(rgb)

        points_list = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            z = depth[v, u]
            if z == 0 or np.isnan(z):
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            # points_list.append([x, y, z, cls_id])  # class_id as fourth channel

            r, g, b = self.class_to_color(cls_id)

            rgb_uint32 = (r << 16) | (g << 8) | b
            rgb_float = struct.unpack('f', struct.pack('I', rgb_uint32))[0]

            points_list.extend(self.create_sphere(x, y, z, rgb_float))

        if points_list:
            points_array = np.array(points_list, dtype=np.float32)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            pc2_msg = point_cloud2.create_cloud(self.cam_info.header, fields, points_array)
            self.pc_pub.publish(pc2_msg)

        # reset buffers to avoid duplicate processing
        self.latest_rgb = None
        self.latest_depth = None

    def class_to_color(self, cls_id):
        colors = [
            (255, 0, 0),    # red
            (0, 255, 0),    # green
            (0, 0, 255),    # blue
            (255, 255, 0),  # yellow
            (255, 0, 255),  # magenta
        ]
        return colors[cls_id % len(colors)]

    def create_sphere(self, x, y, z, rgb_float, radius=0.08, num_points=80):
        pts = []
        for _ in range(num_points):
            dx = np.random.uniform(-radius, radius)
            dy = np.random.uniform(-radius, radius)
            dz = np.random.uniform(-radius, radius)
            pts.append([x + dx, y + dy, z + dz, rgb_float])
        return pts

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
