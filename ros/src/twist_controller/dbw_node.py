#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
       def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        min_speed = rospy.get_param('~min_speed', 0.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # Loop Rate
        self.loop_frequency = 50 # Hz

        # State
        self.dbw_enabled = False
        self.target_angular_velocity = 0.
        self.target_linear_velocity = 0.

        self.current_angular_velocity = 0.
        self.current_linear_velocity = 0.

        # TODO: Create `TwistController` object
        self.controller = Controller( wheel_base,
                                      steer_ratio,
                                      min_speed,
                                      max_lat_accel,
                                      max_steer_angle,
                                      accel_limit,
                                      decel_limit,
                                      self.loop_frequency,
                                      vehicle_mass,
                                      wheel_radius )

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_callback)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_callback)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_callback)

        self.loop()
             
    def velocity_callback(self, msg):
        """
        /current_velocity topic callback handler.
        msg : geometry_msgs.msg.TwistStamped
        Updates state:
        - current_linear_velocity
        - current_angular_velocity
        """
        self.current_linear_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z

    def dbw_enabled_callback(self, msg):
        """
        /vehicle/dbw_enabled topic callback handler.
        msg: Bool indicates if the the car is under drive-by-wire control(True) or if the driver is controlling the car(False)
        Updates state:
        - dbw_enabled
        """
        rospy.loginfo('DBW_ENABLED : {}'.format(msg.data))
        self.dbw_enabled = msg.data

    def twist_callback(self, msg):
        """
        /twist_cmd topic callback handler.
        msg : geometry_msgs.msg.TwistStamped
        Updates state:
        - target_linear_velocity
        - target_angular_velocity
        """
        self.target_linear_velocity = msg.twist.linear.x
        self.target_angular_velocity = msg.twist.angular.z

    def loop(self):
        rate = rospy.Rate(self.loop_frequency)
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            throttle, brake, steer = self.controller.control( self.target_angular_velocity,
                                                                 self.target_linear_velocity,
                                                                 self.current_angular_velocity,
                                                                 self.current_linear_velocity,
                                                                 self.dbw_enabled )
            if self.dbw_enabled:
              self.publish(throttle, brake, steer)
            rate.sleep()
       
    def publish(self, throttle, brake, steer):
        if brake == 0. and throttle >= 0.:
            tcmd = ThrottleCmd()
            tcmd.enable = True
            tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
            tcmd.pedal_cmd = throttle
            self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        if brake >= 0. and throttle == 0.:
            bcmd = BrakeCmd()
            bcmd.enable = True
            bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
            bcmd.pedal_cmd = brake
            self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
