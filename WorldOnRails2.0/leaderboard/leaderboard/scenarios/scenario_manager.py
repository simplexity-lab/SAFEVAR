#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import math
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider
from carla import Vector3D
from srunner.scenariomanager.carla_data_provider import calculate_velocity


class ScenarioManager(object):
    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)
        # --------------add---------------
        self._count = 0
        self.npc_id = None
        self.revert_flag = False
        self.flag_count = 0
        self.distance = 0
        self.overTTC = []
        self.overDRAC = []
        self.t1 = []
        self.t2 = []
        self.index = 0
        self.dac = []
        self.TET = 0
        self.TIT = 0
        self.average_dacc = 0
        self.car_length = 2.4508
        self.npc_x = 210.670166
        # self.record_flag=False
        # --------------------------------

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        # set npc_id
        self.npc_id = scenario.npc_id

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)


    def cal_speed(self, actor):
        velocity_squared = actor.get_velocity().x ** 2
        velocity_squared += actor.get_velocity().y ** 2
        return math.sqrt(velocity_squared)

    def cal_rela_loc(self, actor, pes):
        loc_sq = (actor.get_location().x - pes.get_location().x) ** 2
        loc_sq += (actor.get_location().y - pes.get_location().y) ** 2
        return math.sqrt(loc_sq)

    def cal_rela_speed(self, actor, pes):
        current_dis = actor.get_location().x - self.npc_x
        rela_loc = self.cal_rela_loc(actor, pes)
        cos_rate = current_dis / rela_loc
        # print("cos:",cos_rate)
        v_a = self.cal_speed(actor) * cos_rate
        v_p = self.cal_speed(pes) * math.sqrt(1 - (cos_rate ** 2))
        return v_a + v_p

    def call_TTC(self, actor, pes):
        loc = self.cal_rela_loc(actor, pes)
        velocity = self.cal_rela_speed(actor, pes)
        TTC = (loc - self.car_length) / velocity
        TTC = float('%.3f' % TTC)
        return TTC

    def call_DRAC(self, actor, pes):
        velocity = self.cal_rela_speed(actor, pes)
        loc = self.cal_rela_loc(actor, pes)
        DRAC = (velocity ** 2) / (loc - self.car_length)
        DRAC = float('%.3f' % DRAC)
        return DRAC

    def call_TET(self):
        # Simulator refresh frequency = 0.05
        TET = len(self.overTTC) * 0.05
        return float('%.3f' % TET)

    def call_TIT(self):
        TIT = 0
        for i in range(len(self.overTTC)):
            index = 1.5 - self.overTTC[i]
            TIT = TIT + (index * 0.05)
        return float('%.3f' % TIT)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                ego_action = self._agent()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self.ego_vehicles[0].apply_control(ego_action)

            if ego_action.brake == 1.0:
                if abs(self.ego_vehicles[0].get_velocity().x) > 0.001:
                    self.dac.append(float('%.3f' % self.ego_vehicles[0].get_acceleration().x))

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False
            ego_trans = self.ego_vehicles[0].get_transform()
            # top view : every step

            # set a spectator
            if self._count < 50:
                spectator = CarlaDataProvider.get_world().get_spectator()
                ego_trans = self.ego_vehicles[0].get_transform()
                spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))
                CarlaDataProvider.get_world().debug.draw_string(ego_trans.location, '0', draw_shadow=False,
                                                                color=carla.Color(r=255, g=0, b=0), life_time=100000,
                                                                persistent_lines=True)

            # car head before npc we collect ttc ,if reached do not collect
            if ego_trans.location.x >= 212:
                if float('%.3f' % self.ego_vehicles[0].get_velocity().x) != 0:
                    npc = CarlaDataProvider._carla_actor_pool[self.npc_id]
                    ttc = self.call_TTC(self.ego_vehicles[0], npc)
                    ttc = float('%.3f' % ttc)

                    if abs(ttc) <= 1.5:
                        self.t1.append(ttc)

            # when car stop we stop ttc ,
            if 0.001 > self.ego_vehicles[0].get_velocity().x >= 0:
                self.distance = ego_trans.location.x - self.npc_x - self.car_length
                # 1. car already run 2. record once
                if ego_trans.location.x < 220 and self.index == 0:
                    for i in range(len(self.t1)):
                        self.overTTC.append(self.t1[i])
                    self.index = 1

            # tick count
            self._count = self._count + 1
            if self.npc_id is not None:
                npc = CarlaDataProvider._carla_actor_pool[self.npc_id]
                if npc is not None:
                    control = carla.WalkerControl()
                    control.direction.x = 0
                    control.direction.z = 0
                    control.speed = 1.8
                    if npc.get_location().y > 196:  # how to turn

                        self.revert_flag = True  # to turn
                    if npc.get_location().y < 190:
                        self.revert_flag = False
                    if self.revert_flag:
                        control.direction.y = -1
                    else:
                        control.direction.y = 1
                    npc.apply_control(control)
            # a trick : game over early
            if GameTime.get_time() > 9.5:
                self._running = False

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        self.TET = self.call_TET()
        self.TIT = self.call_TIT()
        total_sum = 0
        for i in range(len(self.dac)):
            total_sum = total_sum + abs(self.dac[i])

        self.average_dacc = float('%.3f' % (total_sum / len(self.dac)))
        # reset variable
        self._count = 0
        self.overTTC = []
        self.t1 = []
        self.t2 = []
        self.index = 0
        self.dac = []

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m' + 'SUCCESS' + '\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        ResultOutputProvider(self, global_result)

    def get_nocrash_diagnostics(self):

        route_completion = None
        lights_ran = None
        duration = round(self.scenario_duration_game, 2)

        for criterion in self.scenario.get_criteria():
            actual_value = criterion.actual_value
            name = criterion.name

            if name == 'RouteCompletionTest':
                route_completion = float(actual_value)
            elif name == 'RunningRedLightTest':
                lights_ran = int(actual_value)

        return route_completion, lights_ran, duration

    def get_nocrash_objective_data(self):
        for criterion in self.scenario.get_criteria():
            name = criterion.name
            if name == 'CollisionTest':
                if criterion.speed is not None:
                    if criterion.speed > 0:
                        speed = - criterion.speed
                        return speed
                    else:
                        return criterion.speed
                if criterion.actual_value == 0:
                    return self.distance

    def get_nocrash_analyze_data(self):
        return self.TET, self.TIT, self.average_dacc
