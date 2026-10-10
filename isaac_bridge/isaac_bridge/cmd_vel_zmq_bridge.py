import pickle
import zmq

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelZMQBridge(Node):

    def __init__(self):
        super().__init__("cmd_vel_zmq_bridge")

        self.zmq_context = zmq.Context()

        self.cmd_socket = self.zmq_context.socket(zmq.PUB)

        # Avoid allowing a large stale command queue.
        self.cmd_socket.setsockopt(zmq.SNDHWM, 1)

        self.cmd_socket.bind("tcp://*:5556")

        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)

        self.get_logger().info("Command bridge ready: /cmd_vel -> tcp://*:5556")

    def cmd_callback(self, msg):

        data = {
            "vx": float(msg.linear.x),
            "vy": float(msg.linear.y),
            "w": float(msg.angular.z),
        }

        self.cmd_socket.send(pickle.dumps(data), flags=zmq.NOBLOCK)


def main(args=None):
    rclpy.init(args=args)

    node = CmdVelZMQBridge()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()