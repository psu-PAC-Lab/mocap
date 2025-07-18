import numpy as np
import quaternion # numpy-quaternion module

import rclpy
from rclpy.node import Node

import nav_msgs.msg
import geometry_msgs.msg
import mocap_msgs.msg

def qdot_2_pqr(q, q_dot):
    q_omega = 2*q.inverse()*q_dot
    omega = np.array([q_omega.x, q_omega.y, q_omega.z])
    return omega

class Mocap(Node):

    def __init__(self):
        super().__init__('mocap')

        # subscribers
        self.rigid_body_sub = self.create_subscription(mocap_msgs.msg.RigidBodies, '/rigid_bodies', self.rigid_bodies_callback, 10)

        # publishers
        self.odom_pub = self.create_publisher(nav_msgs.msg.Odometry, '/nav/odom', 10)

        # parameters
        self.declare_parameter('dpos_mocap2map.x', 0.0)
        self.declare_parameter('dpos_mocap2map.y', 0.0)
        self.declare_parameter('dpos_mocap2map.z', 0.0)
        self.declare_parameter('robot_name', '0')
        self.declare_parameter('filter_time_const', 0.0)
        self.declare_parameter('mocap_freq', 100.0)

        dx = self.get_parameter('dpos_mocap2map.x').get_parameter_value().double_value
        dy = self.get_parameter('dpos_mocap2map.y').get_parameter_value().double_value
        dz = self.get_parameter('dpos_mocap2map.z').get_parameter_value().double_value
        self.dr_mocap2map = np.array([dx, dy, dz])

        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        self.tau_f = self.get_parameter('filter_time_const').get_parameter_value().double_value
        self.dt = 1.0 / self.get_parameter('mocap_freq').get_parameter_value().double_value

        # raw position and quaternion
        self.r_k = None
        self.r_km1 = None
        self.q_k = None
        self.q_km1 = None

        # filtered position and quaternion
        self.rf_k = None
        self.rf_km1 = None
        self.qf_k = None
        self.qf_km1 = None

        # log
        self.get_logger().info(f'dpos_mocap2map: {self.dr_mocap2map}')
        self.get_logger().info(f'robot_name: {self.robot_name}')
        self.get_logger().info(f'tau_f: {self.tau_f}')
        self.get_logger().info(f'dt: {self.dt}')

    def rigid_bodies_callback(self, msg):

        # get data for each rigid body
        for body in msg.rigidbodies:
            if body.rigid_body_name == self.robot_name:

                # read pose in mocap frame
                r_map = np.array([body.pose.position.x, body.pose.position.y, body.pose.position.z])
                self.q_k = np.quaternion(body.pose.orientation.w, body.pose.orientation.x, body.pose.orientation.y, body.pose.orientation.z)

                # convert to map frame
                self.r_k = r_map + self.dr_mocap2map

                pose = geometry_msgs.msg.Pose()
                pose.position.x = self.r_k[0]
                pose.position.y = self.r_k[1]
                pose.position.z = self.r_k[2]
                pose.orientation.x = self.q_k.x
                pose.orientation.y = self.q_k.y
                pose.orientation.z = self.q_k.z
                pose.orientation.w = self.q_k.w

                # if first message, initialize and return
                if self.r_km1 is None:

                    # increment
                    self.r_km1 = self.r_k
                    self.q_km1 = self.q_k
                    self.rf_km1 = self.r_k
                    self.qf_km1 = self.q_k
                    return
                
                # compute twist
                else:
                    twist = self.get_twist()

                # publish odom message
                odom_msg = nav_msgs.msg.Odometry()
                odom_msg.header.stamp = msg.header.stamp
                odom_msg.header.frame_id = 'map'
                odom_msg.child_frame_id = 'base_link'
                odom_msg.pose.pose = pose
                odom_msg.twist.twist = twist
                self.odom_pub.publish(odom_msg)

            else:
                self.get_logger().info(f'Ignoring mocap message for body with name {body.rigid_body_name}')

    # compute twist
    def get_twist(self):

        # first order filtering
        self.rf_k = (1/(2*self.tau_f + self.dt))*((2*self.tau_f-self.dt)*self.rf_km1 + self.dt*(self.r_k + self.r_km1))
        self.qf_k = (1/(2*self.tau_f + self.dt))*((2*self.tau_f-self.dt)*self.qf_km1 + self.dt*(self.q_k + self.q_km1))

        # compute derivatives
        v_k = (self.rf_k - self.rf_km1)/self.dt
        q_dot_k = (self.qf_k - self.qf_km1)/self.dt

        # compute linear velocity
        q_v = np.quaternion(0, v_k[0], v_k[1], v_k[2])
        q_vb = self.q_k.inverse()*q_v*self.q_k
        v = np.array([q_vb.x, q_vb.y, q_vb.z])
        
        # angular velocity
        omega = qdot_2_pqr(self.qf_k, q_dot_k)

        # increment
        self.r_km1 = self.r_k
        self.q_km1 = self.q_k
        self.rf_km1 = self.rf_k
        self.qf_km1 = self.qf_k

        # twist
        twist = geometry_msgs.msg.Twist()
        twist.linear.x = v[0]
        twist.linear.y = v[1]
        twist.linear.z = v[2]
        twist.angular.x = omega[0]
        twist.angular.y = omega[1]
        twist.angular.z = omega[2]
        return twist

def main(args=None):
    rclpy.init(args=args)
    mocap_node = Mocap()
    rclpy.spin(mocap_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()