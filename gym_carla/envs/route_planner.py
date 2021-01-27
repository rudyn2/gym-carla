#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This file is modified from <https://github.com/carla-simulator/carla>:
# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from enum import Enum
from collections import deque
import random
import numpy as np
import xmltodict
import carla
import logging

from gym_carla.envs.misc import distance_vehicle, is_within_distance_ahead, compute_magnitude_angle
from leaderboard.utils.route_manipulation import interpolate_trajectory

logger = logging.root


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4


class RoutePlanner:
    def __init__(self, vehicle, buffer_size, route_path):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._route_path = route_path

        self._min_distance = 4

        self._target_waypoint = None
        self._buffer_size = buffer_size

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._target_road_option = RoadOption.LANEFOLLOW

        self._last_traffic_light = None
        self._proximity_threshold = 15.0

        self._waypoint_buffer = self._load_route(self._world, self._route_path, self._buffer_size)

    @staticmethod
    def _load_route(world, route_path, buffer_size):
        map = world.get_map()
        waypoint_buffer = deque(maxlen=buffer_size)
        with open(route_path, "r") as xml_obj:
            routes_xml = xmltodict.parse(xml_obj.read())['routes']['route']
            route = routes_xml[0]
            trajectory = []
            for waypoint in route['waypoint']:
                loc = carla.Location(x=float(waypoint['@x']),
                                     y=float(waypoint['@y']),
                                     z=float(waypoint['@z']))
                trajectory.append(loc)
            new_trajectory = interpolate_trajectory(world, trajectory)

            # load buffer of waypoints
            for transform, road_option in new_trajectory[1]:
                wp = map.get_waypoint(transform.location)
                waypoint_buffer.append((wp, road_option))
        return waypoint_buffer

    @staticmethod
    def get_init_pos(world, route_path):
        buffer = RoutePlanner._load_route(world, route_path, 5000)
        waypoint, _ = buffer[0]
        return waypoint

    def run_step(self):
        waypoints = self._get_waypoints()
        red_light, vehicle_front = self._get_hazard()
        # red_light = False
        return waypoints, red_light, vehicle_front

    def _get_waypoints(self):
        """
        Loads route stored waypoints
        :return:
        """
        waypoints = []
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            waypoints.append(
                [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.rotation.yaw])

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self._target_waypoint, self._target_road_option = self._waypoint_buffer[-1]

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        # remove past waypoints
        if max_index >= 0:
            for _ in range(max_index - 1):
                self._waypoint_buffer.popleft()

        return waypoints

    def _get_hazard(self):
        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state = self._is_vehicle_hazard(vehicle_list)

        # check for the state of the traffic lights
        light_state = self._is_light_red_us_style(lights_list)

        return light_state, vehicle_state

    def _is_vehicle_hazard(self, vehicle_list):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
             - bool_flag is True if there is a vehicle ahead blocking us
               and False otherwise
             - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = target_vehicle.get_location()
            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold):
                return True

        return False

    def _is_light_red_us_style(self, lights_list):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
             - bool_flag is True if there is a traffic light in RED
               affecting us and False otherwise
             - traffic_light is the object itself or None if there is no
               red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_intersection:
            # It is too late. Do not block the intersection! Keep going!
            return False

        if self._target_waypoint is not None:
            if self._target_waypoint.is_intersection:
                potential_lights = []
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 80.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.libcarla.TrafficLightState.Red:
                        return True
                else:
                    self._last_traffic_light = None

        return False


def retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
         candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
         RoadOption.STRAIGHT
         RoadOption.LEFT
         RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
