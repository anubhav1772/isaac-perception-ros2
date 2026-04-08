import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
from object_detection.yolo_detector import YOLOv8Detector
import struct
import tf2_ros
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from std_msgs.msg import Header

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()
        self.detector = YOLOv8Detector(self, "yolov8n.pt")  # use your trained model

        # subscribers
        self.rgb_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/rgb/camera_info', self.cam_info_callback, 10)

        # publishers
        self.marker_pub = self.create_publisher(MarkerArray, "/detected_objects_markers", 10)
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

    def publish_markers(self, detections, header):
        print("PUBLISH MARKERS CALLED, count:", len(detections))
        marker_array = MarkerArray()

        for i, det in enumerate(detections):

            x, y, z = det["pos"]
            label = det["label"]

            # CUBE (bounding proxy)
            cube = Marker()
            # cube.header = header
            cube.header.frame_id = "odom"
            cube.header.stamp = self.get_clock().now().to_msg()
            cube.ns = "objects"
            cube.id = i
            cube.type = Marker.CUBE
            cube.action = Marker.ADD

            cube.pose.position.x = x
            cube.pose.position.y = y
            cube.pose.position.z = z

            cube.pose.orientation.w = 1.0

            cube.scale.x = 0.3
            cube.scale.y = 0.3
            cube.scale.z = 0.3

            cube.color.r = 1.0
            cube.color.g = 0.0
            cube.color.b = 0.0
            cube.color.a = 0.8

            cube.lifetime.sec = 1

            marker_array.markers.append(cube)

            # TEXT LABEL
            text = Marker()
            # text.header = header
            text.header.frame_id = "odom"
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "labels"
            text.id = i + 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + 0.4 # above object

            text.pose.orientation.w = 1.0

            text.scale.z = 0.3
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            text.text = label

            text.lifetime.sec = 1

            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

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

        detections_out = []

        points_list = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            # Single point/pixel
            # z = depth[v, u]
            # if z == 0 or np.isnan(z):
            #     continue

            # Median depth in a small patch
            # Instead of 1 pixel -> use a small window around (u, v)
            # patch_size = 5
            # half = patch_size // 2
            #
            # u_min = max(u - half, 0)
            # u_max = min(u + half, depth.shape[1] - 1)
            # v_min = max(v - half, 0)
            # v_max = min(v + half, depth.shape[0] - 1)
            #
            # patch = depth[v_min:v_max, u_min:u_max]
            #
            # # remove invalid values
            # valid = patch[(patch > 0.1) & np.isfinite(patch)]
            #
            # if len(valid) == 0:
            #     continue
            #
            # z = np.median(valid)

            # Take multiple samples inside bbox
            samples = []
            for _ in range(15):
                uu = np.random.randint(int(x1), int(x2))
                vv = np.random.randint(int(y1), int(y2))

                if 0 <= vv < depth.shape[0] and 0 <= uu < depth.shape[1]:
                    z_val = depth[vv, uu]

                    if z_val > 0.1 and np.isfinite(z_val):
                        samples.append(z_val)

            if len(samples) == 0:
                continue

            z = np.median(samples)

            # x = (u - cx) * z / fx
            # y = (v - cy) * z / fy
            # # points_list.append([x, y, z, cls_id])  # class_id as fourth channel
            #
            # r, g, b = self.class_to_color(cls_id)
            #
            # rgb_uint32 = (r << 16) | (g << 8) | b
            # rgb_float = struct.unpack('f', struct.pack('I', rgb_uint32))[0]
            #
            # points_list.extend(self.create_sphere(x, y, z, rgb_float))

            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            z_cam = z

            # Create point in camera frame
            point_cam = PointStamped()
            point_cam.header.frame_id = "camera_optical_frame"
            point_cam.header.stamp = self.get_clock().now().to_msg()

            point_cam.point.x = float(x_cam)
            point_cam.point.y = float(y_cam)
            point_cam.point.z = float(z_cam)

            try:
                # Transform to odom/world frame
                point_world = self.tf_buffer.transform(point_cam, "odom")

                x = point_world.point.x
                y = point_world.point.y
                z = point_world.point.z

            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue

            class_name = self.detector.model.names[cls_id]
            detections_out.append({
                "pos": (
                    x,
                    y,
                    z
                ),
                "label": class_name,
                "id": cls_id
            })

            # Color
            r, g, b = self.class_to_color(cls_id)
            rgb_uint32 = (r << 16) | (g << 8) | b
            rgb_float = struct.unpack('f', struct.pack('I', rgb_uint32))[0]

            points_list.extend(self.create_sphere(x, y, z, rgb_float))

        if detections_out:
            # header = self.cam_info.header

            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "odom"

            # publish markers
            self.publish_markers(detections_out, header)

        if points_list:
            points_array = np.array(points_list, dtype=np.float32)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            # frame_id = camera_optical_frame
            # pc2_msg = point_cloud2.create_cloud(self.cam_info.header, fields, points_array)

            header = self.cam_info.header
            header.frame_id = "odom"

            pc2_msg = point_cloud2.create_cloud(header, fields, points_array)
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
