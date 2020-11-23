#!/usr/bin/env python

import rospy

from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32, Bool
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
NO_TRAFFIC_LIGHT = -1



class WaypointUpdater(object):
    def __init__(self):
        self.tl_detector_initialized = False

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_callback)
        rospy.Subscriber('/tl_detector_initialized', Bool, self.tl_detector_initialized_cb)
        self.base_waypoint_subscriber = rospy.Subscriber('/base_wp', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_wp_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_wp = None
        self.base_wps_count = None
        self.current_pose = None
        
        self.loop_frequency = 2 # Loop_Rate_Hz
        self.max_velocity = None  # Max_Velocity_miles/sec
        self.traffic_light_index = NO_TRAFFIC_LIGHT  #Traffic_light_Index_no_raffic_light.
        self.current_velocity = 0. # Current_velocity
        self.loop()

        rospy.spin()


    def loop(self):

        rate = rospy.Rate(self.loop_frequency)
        while not rospy.is_shutdown():
            if not self.tl_detector_initialized:
                rospy.logwarn('WP_Updater_Holding')
                rate.sleep()
                continue

            if self.current_pose != None and self.base_wp != None:
                xyz_position = self.current_pose.position
                quaternion_orientation = self.current_pose.orientation
                p = xyz_position
                move = quaternion_orientation
                p_list = [p.x, p.y, p.z]
                move_list = [move.x, move.y, move.z, move.w]
                euler = euler_from_quaternion(move_list)
                yaw_rad = euler[2]

                closest_wp_indx = None
                closest_wp_dist = None
                for idx in range(len(self.base_wp)):
                    wcx, wcy = self.get_on_car_waypoint_x_y(p, yaw_rad, idx)
                    if closest_wp_indx is None:
                        closest_wp_indx = idx
                        closest_wp_dist = math.sqrt(wcx**2 + wcy**2)
                    else:
                        current_wp_dist = math.sqrt(wcx**2 + wcy**2)
                        if current_wp_dist < closest_wp_dist: #
                            closest_wp_indx = idx
                            closest_wp_dist = current_wp_dist



                wcx, wcy = self.get_on_car_waypoint_x_y(p, yaw_rad, closest_wp_indx)
                while wcx < 0.:
                    closest_wp_indx = (closest_wp_indx + 1) % self.base_wps_count
                    wcx, wcy = self.get_on_car_waypoint_x_y(p, yaw_rad, closest_wp_indx)

                next_waypoints = []
                for loop_idx in range(LOOKAHEAD_WPS):
                    wp_idx = (loop_idx + closest_wp_indx) % self.base_wps_count
                    next_waypoints.append(self.get_waypoint_to_sent(wp_idx))

                rospy.loginfo('INDEX {} wc [{:.3f},{:.3f}]'.format(closest_wp_indx, wcx, wcy))
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)
                lane.waypoints = self.adjust_velocity_to_stop(next_waypoints, closest_wp_indx)
                self.final_wp_pub.publish(lane)


            rate.sleep()

    def adjust_velocity_to_stop(self, waypoints, closest_wp_indx):
        traffic_index = self.traffic_light_index
        if traffic_index == NO_TRAFFIC_LIGHT:
            rospy.loginfo('TRAFFIC no traffic lights ahead')
            return waypoints

        # Map traffic index to current list
        if traffic_index < closest_wp_indx:
            traffic_index = self.base_wps_count - closest_wp_indx + traffic_index
        else:
            traffic_index = traffic_index - closest_wp_indx

        if traffic_index >= LOOKAHEAD_WPS:
            rospy.loginfo('TRAFFIC no traffic lights before LOOKAHEAD_WPS {}'.format(traffic_index))
            return waypoints

        v_zero = self.current_velocity

        if v_zero < 1.:
            min_distance_to_stop = 1.
            max_distance_to_stop = 5.
        else:
            min_distance_to_stop = 1.5 * v_zero
            max_distance_to_stop = 2. * min_distance_to_stop

        distance2stop = self.distance(waypoints, 0, traffic_index)
        if distance2stop < min_distance_to_stop:
            distance2stop = 0
        else:
            distance2stop -= min_distance_to_stop

        if distance2stop < 0.0001:
            m = 0.
        else:
            m = v_zero / distance2stop

        for index in range(traffic_index):
            distance = self.distance(waypoints, 0, index)
            from_stop_distance = distance2stop - distance
            if from_stop_distance <= min_distance_to_stop:
                velocity = 0
            else:
                if from_stop_distance > max_distance_to_stop:
                    velocity = self.max_velocity
                else:
                    velocity = v_zero - m * distance
            self.set_waypoint_velocity(waypoints, index , velocity)
            
        for index in range(traffic_index, LOOKAHEAD_WPS):
            self.set_waypoint_velocity(waypoints, index , 0)


        rospy.loginfo('TRAFFIC traffic lights stop at {}, distance to stop : {:.3f} waypoints with zero {}'.format(traffic_index, distance2stop, LOOKAHEAD_WPS - traffic_index))
        return waypoints

    def get_on_car_waypoint_x_y(self, current_possition, yaw_rad, index):
        wgx, wgy = self.get_waypoint_x_y(index)
        return get_car_xy_from_global_xy(current_possition.x, current_possition.y, yaw_rad, wgx, wgy)

    def get_waypoint_x_y(self, index):
        waypoint = self.base_wp[index]
        x = waypoint.pose.pose.position.x
        y = waypoint.pose.pose.position.y
        return x, y

    def get_waypoint_to_sent(self, wp_idx):
        self.set_waypoint_velocity(self.base_wp, wp_idx, self.max_velocity)
        return self.base_wp[wp_idx]

    def tl_detector_initialized_cb(self, msg):
        self.tl_detector_initialized = True

    def pose_cb(self, msg):
        self.current_pose = msg.pose

    def waypoints_cb(self, lane):
        if self.base_wp != None:
            return

        self.base_waypoint_subscriber.unregister()
        self.base_waypoint_subscriber = None

        self.max_velocity = max([self.get_waypoint_velocity(wp) for wp in lane.waypoints]) * .99
        rospy.loginfo('MAX_VELOCITY : {}'.format(self.max_velocity))

        self.base_wps_count = len(lane.waypoints)
        self.base_wp = lane.waypoints


    def traffic_cb(self, msg):
        self.traffic_light_index = msg.data;
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def velocity_callback(self, msg):
        self.current_velocity = msg.twist.linear.x

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')


def get_car_xy_from_global_xy(car_x, car_y, yaw_rad, global_x, global_y):
    # Translate global point by car's position
    xg_trans_c = global_x - car_x
    yg_trans_c = global_y - car_y
    # Perform rotation to finish mapping
    # from global coords to car coords
    x = xg_trans_c * math.cos(0 - yaw_rad) - yg_trans_c * math.sin(0 - yaw_rad)
    y = xg_trans_c * math.sin(0 - yaw_rad) + yg_trans_c * math.cos(0 - yaw_rad)
    return (x, y)
