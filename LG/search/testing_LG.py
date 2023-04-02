from environs import Env
import time
import os
import lgsvl
from lgsvl.geometry import Vector, Transform
from prettytable import PrettyTable


class LG:
    def __init__(self):
        self.env = Env()

        self.sim = lgsvl.Simulator(self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host),
                                   self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        self.ego = None
        self.spawns = None
        self.npc = None

    def destory(self):
        self.ego = None
        self.spawns = None
        self.npc = None

    def create_ego_and_connect(self, start, target):
        table = PrettyTable()
        if self.sim.current_scene == lgsvl.wise.DefaultAssets.map_borregasave:
            self.sim.reset()
        else:
            self.sim.load(lgsvl.wise.DefaultAssets.map_borregasave)
        self.spawns = self.sim.get_spawn()

        """
         set ego location behind light
        """
        state = lgsvl.AgentState()
        state.transform = self.spawns[0]

        """   before light
        layer_mask = 0
        layer_mask |= 1 << 0
        # ego
        state = lgsvl.AgentState()
        forward = lgsvl.utils.transform_to_forward(self.spawns[0])
        state.transform.position = self.spawns[0].position + 40 * forward
        state.transform.rotation = self.spawns[0].rotation
        # print("sp:", self.spawns[0].position + 40 * forward)
        hit = self.sim.raycast(self.spawns[0].position + 40 * forward, lgsvl.Vector(0, -1, 0),
                          layer_mask)
        # # Agents can be spawned with a velocity. Default is to spawn with 0 velocity

        state.transform.position = hit.point
        # print("hit:", hit.point)
        state.velocity = 20 * forward
        
        """

        self.ego = self.sim.add_agent(name="511086bd-97ad-4109-b0ad-654ba662fbcf", agent_type=lgsvl.AgentType.EGO,
                                      state=state)

        self.ego.connect_bridge(self.env.str("LGSVL__AUTOPILOT_0_HOST", lgsvl.wise.SimulatorSettings.bridge_host),
                                self.env.int("LGSVL__AUTOPILOT_0_PORT", lgsvl.wise.SimulatorSettings.bridge_port))

        # bridge connection
        print("Waiting for connection...")
        while not self.ego.bridge_connected:
            time.sleep(1)
        print("Get connection!")

        # get current parameter
        s = self.ego.state
        if s is not None:
            table.add_column('参数',
                             ['angular', 'speed', 'tip', 'mass', 'wheel_mass', 'wheel_radius', 'MaxMotorTorque',
                              'MaxBrakeTorque', 'MinRPM', 'MaxRPM', 'ShiftTime', 'TireDragCoeff', 'WheelDamping',
                              'TractionControlSlipLimit', 'MaxSteeringAngle', 'position'])
            table.add_column('值',
                             [s.angular_velocity, s.velocity, s.tip, s.mass, s.wheel_mass, s.wheel_radius,
                              s.MaxMotorTorque,
                              s.MaxBrakeTorque, s.MinRPM, s.MaxRPM, s.ShiftTime, s.TireDragCoeff, s.WheelDamping,
                              s.TractionControlSlipLimit, s.MaxSteeringAngle, s.position])
            print(table)

        # Dreamview setup
        dv = lgsvl.dreamview.Connection(self.sim, self.ego, self.env.str("LGSVL__AUTOPILOT_0_HOST", "127.0.0.1"))
        dv.set_hd_map('Borregas Ave')
        dv.set_vehicle('Lincoln2017MKZ')
        modules = [
            'Localization',
            'Perception',
            'Transform',
            'Routing',
            'Prediction',
            'Planning',
            'Camera',
            'Traffic Light',
            'Control'
        ]

        # reset destination
        destination = self.spawns[0].destinations[0]
        dv.setup_apollo(destination.position.x, destination.position.z, modules)

    def set_NPC_stop(self):
        # npc stop 3.25--------------------------------------

        state_npc = lgsvl.AgentState()
        state_npc.transform = self.spawns[0]
        state_npc.transform.position = Vector(165.646987915039, -4.49, -66)
        npc = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state_npc)

        # ------------------------------------------------------

    def set_Pedestrian(self):
        # ---------pes-set--------before run----------------------------
        state_p = lgsvl.AgentState()
        state_p.transform = self.spawns[0]
        tr = Transform(state_p.transform)
        tr.position = Vector(165.646987915039, -4.49, -70)  # right

        wp = []
        wp.append(lgsvl.WalkWaypoint(tr.position, idle=0, speed=2))
        tr.position = Vector(165.646987915039, -4.49, -63)  # left
        wp.append(lgsvl.WalkWaypoint(tr.position, idle=0, speed=2))
        state_p.transform.position = wp[0].position
        p = self.sim.add_agent("Pamela", lgsvl.AgentType.PEDESTRIAN, state_p)

        def on_waypoint_1(agent, index):
            print("Waypoint {} reached".format(index))
        p.on_waypoint_reached(on_waypoint_1)
        p.follow(wp, True)



    def set_npc_follow(self):
        # -----------------------------------------------------------------
        # npc follow ------3.28--3.29------------------------------------------------\
        layer_mask = 0
        layer_mask |= 1 << 0
        waypoints = []
        forward = lgsvl.utils.transform_to_forward(self.spawns[0])

        """
            set npc location
        """
        state_npc = lgsvl.AgentState()
        state_npc.transform.position = self.spawns[0].position + forward * 60
        state_npc.transform.rotation = self.spawns[0].rotation
        hit = self.sim.raycast(self.spawns[0].position + forward * 60, lgsvl.Vector(0, -1, 0),
                               layer_mask)

        state_npc.transform.position = hit.point

        self.npc = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state_npc)
        wp = lgsvl.DriveWaypoint(
            position=state_npc.transform.position, speed=20, angle=self.spawns[0].rotation, idle=0, trigger_distance=10
        )
        waypoints.append(wp)

        z_delta = 32
        pz = 0
        speed_tmp = 16
        for i in range(5):
            temp = i + 1
            # temp = 1 / temp
            # pz = temp * z_delta
            pz = pz + z_delta
            z_delta = z_delta / 2
            speed = speed_tmp
            if speed_tmp > 2:
                speed_tmp = speed_tmp / 2

            angle = self.spawns[0].rotation

            hit = self.sim.raycast(self.spawns[0].position + forward * 60 + forward * pz, lgsvl.Vector(0, -1, 0),
                                   layer_mask)
            # print(pz)
            # print(i, "sp", spawns[0].position + forward * 60)
            # print(i, "pz", spawns[0].position + forward * 60 + forward * pz)
            # print(i, "hit:", hit.point)

            wp = lgsvl.DriveWaypoint(
                position=hit.point, speed=speed, angle=angle, idle=0, trigger_distance=0
            )
            waypoints.append(wp)

        def on_waypoint(agent, index):
            print("waypoint {} reached, waiting for ego to get closer".format(index))
        self.npc.on_waypoint_reached(on_waypoint)

        self.npc.follow(waypoints)
        # --------------------------------------------------------------



    def set_physics(self, phy):
        # 3.25 : phy -----0.35 -NO--------before run------change parameter-----4.6--
        # phy = [2120, 30, 0.35, 450, 3000, 800, 8299, 0.4, 4, 1, 0.8, 39.4]

        # phy = [2120, 30, 0.35, 8299, 800, 3000, 450, 39.4, 4, 1, 0.4, 0.8]
        x = self.ego.state
        x.mass = phy[0]
        x.wheel_mass = phy[1]
        x.wheel_radius = phy[2]
        x.MaxRPM = phy[3]
        x.MinRPM = phy[4]
        x.MaxBrakeTorque = phy[5]
        x.MaxMotorTorque = phy[6]
        x.MaxSteeringAngle = phy[7]
        x.TireDragCoeff = phy[8]
        x.WheelDamping = phy[9]
        x.ShiftTime = phy[10]
        x.TractionControlSlipLimit = phy[11]
        self.ego.state = x
        # ----------------------------------------------------------------------

    def run(self):
        self.sim.run()
        result = self.npc.state.transform.position -self.ego.state.transform.position
        print("distance:", result)
        return result


    def get_physics(self):
        s = self.ego.state
        table = PrettyTable()
        if s is not None:
            table.add_column('参数', ['angular', 'speed', 'tip', 'mass', 'wheel_mass', 'wheel_radius', 'MaxMotorTorque',
                                    'MaxBrakeTorque', 'MinRPM', 'MaxRPM', 'ShiftTime', 'TireDragCoeff', 'WheelDamping',
                                    'TractionControlSlipLimit', 'MaxSteeringAngle', 'position'])
            table.add_column('值',
                             [s.angular_velocity, s.velocity, s.tip, s.mass, s.wheel_mass, s.wheel_radius,
                              s.MaxMotorTorque,
                              s.MaxBrakeTorque, s.MinRPM, s.MaxRPM, s.ShiftTime, s.TireDragCoeff, s.WheelDamping,
                              s.TractionControlSlipLimit, s.MaxSteeringAngle, s.position])
            print(table)
        return table

# 3.25 -------------------------------------------------------
# print(state_npc.transform.position - ego.state.position)
# ------------------------------------------------------------
