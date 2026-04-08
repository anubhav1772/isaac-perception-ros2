import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster
import struct

import zmq
import pickle
import numpy as np


class ZMQBridge(Node):

    def __init__(self):
        super().__init__('isaac_zmq_bridge')

        # ROS publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)

        # Ground-truth / perfect pose
        self.odom_pub = self.create_publisher(Odometry, '/ground_truth/odom', 10)
        self.imu_pub = self.create_publisher(Imu, '/camera/imu', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/camera/points', 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, '/camera/rgb/camera_info', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # self.publish_camera_tf()
        # self.publish_imu_tf()

        # DEBUG via: ros2 topic echo /tf_static
        # MUST contain base_link -> camera_link & camera_link -> camera_imu_frame
        self.publish_static_transforms()
        self.initial_pose = None

        self.bridge = CvBridge()

        # ZMQ subscriber
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # timer to poll ZMQ
        # self.timer = self.create_timer(0.03, self.receive_data)  # ~30 Hz
        self.timer = self.create_timer(0.1, self.receive_data)  # ~10 Hz

    # def publish_camera_tf(self):
    #     t = TransformStamped()
    #
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = "base_link"
    #     t.child_frame_id = "camera_link"
    #
    #     # MUST match Isaac Gym camera placement
    #     t.transform.translation.x = 0.3
    #     t.transform.translation.y = 0.0
    #     t.transform.translation.z = 0.2
    #
    #     # Keep identity (you said 0,0,0 works)
    #     t.transform.rotation.w = 1.0
    #
    #     # self.static_tf_broadcaster.sendTransform(t)
    #     # send multiple times
    #     for _ in range(5):
    #         self.static_tf_broadcaster.sendTransform(t)
    #
    # def publish_imu_tf(self):
    #     t = TransformStamped()
    #
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = "camera_link"
    #     t.child_frame_id = "camera_imu_frame"
    #
    #     # IMU is usually at camera center (or very close)
    #     t.transform.translation.x = 0.0
    #     t.transform.translation.y = 0.0
    #     t.transform.translation.z = 0.0
    #
    #     # Identity rotation
    #     t.transform.rotation.w = 1.0
    #
    #     self.static_tf_broadcaster.sendTransform(t)

    def publish_static_transforms(self):
        transforms = []

        # now = self.get_clock().now().to_msg()

        # base_link -> camera_link
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = "base_link"
        t1.child_frame_id = "camera_link"

        t1.transform.translation.x = 0.3
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.2
        t1.transform.rotation.w = 1.0

        transforms.append(t1)

        # camera_link -> camera_imu_frame
        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = "camera_link"
        t2.child_frame_id = "camera_imu_frame"

        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0
        t2.transform.rotation.w = 1.0

        transforms.append(t2)

        import tf_transformations

        t3 = TransformStamped()

        t3.header.stamp = self.get_clock().now().to_msg()
        t3.header.frame_id = "camera_link"
        t3.child_frame_id = "camera_optical_frame"

        # no translation
        t3.transform.translation.x = 0.0
        t3.transform.translation.y = 0.0
        t3.transform.translation.z = 0.0

        # KEY ROTATION (ROS standard)
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
            -np.pi/2, 0, -np.pi/2
        )

        t3.transform.rotation.x = qx
        t3.transform.rotation.y = qy
        t3.transform.rotation.z = qz
        t3.transform.rotation.w = qw

        transforms.append(t3)

        self.static_tf_broadcaster.sendTransform(transforms)

    def create_pointcloud2(self, depth, intrinsics, stamp):

        H, W = depth.shape

        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]

        points = []

        for v in range(H):
            for u in range(W):
                z = depth[v, u]
                if z == 0:
                    continue

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                points.append([x, y, z])

        # Create PointCloud2
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_link"

        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        buffer = []
        for p in points:
            buffer.append(struct.pack('fff', p[0], p[1], p[2]))

        msg.data = b''.join(buffer)

        return msg

    def depth_to_pc_fast(self, depth, rgb, intrinsics, stamp):

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        H, W = depth.shape
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        u, v = np.meshgrid(np.arange(W), np.arange(H))

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # valid = (z > 0) & np.isfinite(z)
        # Range filter (remove far noise)
        valid = (z > 0.1) & (z < 5.0) & np.isfinite(z)

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        # filter valid points
        points = points[valid.reshape(-1)]

        colors = rgb.reshape(-1, 3)
        colors = colors[valid.reshape(-1)]

        # downsampling (Take every 4th point)
        points = points[::4]
        colors = colors[::4]
        # colors = colors[:, ::-1]  # BGR -> RGB

        rgb_uint32 = (
            (colors[:, 0].astype(np.uint32) << 16) |
            (colors[:, 1].astype(np.uint32) << 8)  |
            (colors[:, 2].astype(np.uint32))
        )

        rgb_float = rgb_uint32.view(np.float32)

        cloud = np.zeros(points.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32)
        ])

        cloud['x'] = points[:, 0]
        cloud['y'] = points[:, 1]
        cloud['z'] = points[:, 2]
        cloud['rgb'] = rgb_float

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_optical_frame"

        msg.height = 1
        msg.width = cloud.shape[0]

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * cloud.shape[0]
        msg.is_dense = True

        msg.data = cloud.tobytes()
        return msg

    def receive_data(self):
        try:
            msg = self.socket.recv(flags=zmq.NOBLOCK)
            data = pickle.loads(msg)

            now = self.get_clock().now().to_msg()

            # RGB + DEPTH
            rgb = data["rgb"]
            depth = data["depth"]

            rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')

            rgb_msg.header.stamp = now
            depth_msg.header.stamp = now

            rgb_msg.header.frame_id = "camera_optical_frame"
            depth_msg.header.frame_id = "camera_optical_frame"

            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)

            # IMU
            imu_data = data.get("imu", None)

            if imu_data is not None:
                imu_msg = Imu()

                imu_msg.header.stamp = now
                imu_msg.header.frame_id = "camera_imu_frame"

                # orientation
                imu_msg.orientation.x = float(imu_data["quat"][0])
                imu_msg.orientation.y = float(imu_data["quat"][1])
                imu_msg.orientation.z = float(imu_data["quat"][2])
                imu_msg.orientation.w = float(imu_data["quat"][3])

                # # angular velocity
                # imu_msg.angular_velocity.x = float(imu_data["ang_vel"][0])
                # imu_msg.angular_velocity.y = float(imu_data["ang_vel"][1])
                # imu_msg.angular_velocity.z = float(imu_data["ang_vel"][2])

                # with balanced noise
                imu_msg.angular_velocity.x = float(imu_data["ang_vel"][0]) + np.random.normal(0, 0.01)
                imu_msg.angular_velocity.y = float(imu_data["ang_vel"][1]) + np.random.normal(0, 0.01)
                imu_msg.angular_velocity.z = float(imu_data["ang_vel"][2]) + np.random.normal(0, 0.01)

                # # linear acceleration
                # imu_msg.linear_acceleration.x = float(imu_data["lin_acc"][0])
                # imu_msg.linear_acceleration.y = float(imu_data["lin_acc"][1])
                # imu_msg.linear_acceleration.z = float(imu_data["lin_acc"][2])

                imu_msg.linear_acceleration.x = float(imu_data["lin_acc"][0]) + np.random.normal(0, 0.1)
                imu_msg.linear_acceleration.y = float(imu_data["lin_acc"][1]) + np.random.normal(0, 0.1)
                imu_msg.linear_acceleration.z = float(imu_data["lin_acc"][2]) + np.random.normal(0, 0.1)

                imu_msg.orientation_covariance[0] = -1
                imu_msg.angular_velocity_covariance[0] = -1
                imu_msg.linear_acceleration_covariance[0] = -1

                self.imu_pub.publish(imu_msg)

            # ODOM (GROUND TRUTH)
            pose = data["pose"]

            p = pose["position"]

            # store initial pose once
            if self.initial_pose is None:
                self.initial_pose = p

            # subtract initial offset
            x = float(p[0] - self.initial_pose[0])
            y = float(p[1] - self.initial_pose[1])

            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"

            # position
            # odom.pose.pose.position.x = float(pose["position"][0])
            # odom.pose.pose.position.y = float(pose["position"][1])
            # odom.pose.pose.position.z = float(pose["position"][2]) # ~ 0.30 (base of aliengo)

            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.position.z = 0.0

            # orientation
            odom.pose.pose.orientation.x = float(pose["orientation"][0])
            odom.pose.pose.orientation.y = float(pose["orientation"][1])
            odom.pose.pose.orientation.z = float(pose["orientation"][2])
            odom.pose.pose.orientation.w = float(pose["orientation"][3])

            # velocity
            odom.twist.twist.linear.x = float(pose["linear_velocity"][0])
            odom.twist.twist.linear.y = float(pose["linear_velocity"][1])
            odom.twist.twist.linear.z = float(pose["linear_velocity"][2])

            odom.twist.twist.angular.x = float(pose["angular_velocity"][0])
            odom.twist.twist.angular.y = float(pose["angular_velocity"][1])
            odom.twist.twist.angular.z = float(pose["angular_velocity"][2])

            self.odom_pub.publish(odom)

            # TF: odom -> base_link
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "odom"
            t.child_frame_id = "base_link"

            # t.transform.translation.x = float(pose["position"][0])
            # t.transform.translation.y = float(pose["position"][1])
            # t.transform.translation.z = float(pose["position"][2])

            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = 0.0

            t.transform.rotation.x = float(pose["orientation"][0])
            t.transform.rotation.y = float(pose["orientation"][1])
            t.transform.rotation.z = float(pose["orientation"][2])
            t.transform.rotation.w = float(pose["orientation"][3])

            self.tf_broadcaster.sendTransform(t)

            intrinsics = data["intrinsics"]

            pc_msg = self.depth_to_pc_fast(depth, rgb, intrinsics, now)
            self.pc_pub.publish(pc_msg)

            cam_info = CameraInfo()
            cam_info.header.stamp = now
            cam_info.header.frame_id = "camera_optical_frame"

            cam_info.width = 640
            cam_info.height = 480

            fx = intrinsics["fx"]
            fy = intrinsics["fy"]
            cx = intrinsics["cx"]
            cy = intrinsics["cy"]

            cam_info.k = [
                float(fx), 0.0, float(cx),
                0.0, float(fy), float(cy),
                0.0, 0.0, 1.0
            ]

            cam_info.p = [float(fx), 0.0, float(cx), 0.0,
                          0.0, float(fy), float(cy), 0.0,
                          0.0, 0.0, 1.0, 0.0]

            self.cam_info_pub.publish(cam_info)

        except zmq.Again:
            pass

def main():
    rclpy.init()
    node = ZMQBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
