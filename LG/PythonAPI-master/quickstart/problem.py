import random
import numpy as np
# import simplexity_carla_combination as cc
import again as lg
from jmetal.core.problem import FloatProblem, BinaryProblem, Problem
from jmetal.core.solution import FloatSolution, BinarySolution, IntegerSolution, CompositeSolution
import lgsvl
from lgsvl.geometry import Vector
from environs import Env
import time


class LGSVLProblem(FloatProblem):

    def __init__(self, env, sim,ss):
        super(LGSVLProblem, self).__init__()
        self.number_of_variables = 12
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_direction = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(1)', 'f(2)', 'f(3)']

        self.lower_bound = [2000, 20, 0.30, 6000, 600, 2500, 400, 30, 2, 0.15, 0.2, 0.65]
        self.upper_bound = [2500, 60, 0.39, 13000, 1100, 3150, 550, 50, 5, 1.5, 0.6, 0.95]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        self.env = env
        self.sim = sim
        self.npc = None
        self.sim.load(lgsvl.wise.DefaultAssets.map_borregasave)
        # self.sim.weather = lgsvl.WeatherState(rain=0.9, fog=0.9, wetness=0.9, cloudiness=0.9, damage=0)
        self.spawns = self.sim.get_spawn()
        state_npc = lgsvl.AgentState()
        state_npc.transform = self.spawns[0]
        state_npc.transform.position = Vector(165.646987915039, -4.49, -63)
        self.npc = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state_npc)
        state_npc.transform.position = Vector(165.646987915039, -4.49, -60)
        npc_1 = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state_npc)
        self.ego = None
        self.ss = ss
        self.temp = []
        self.restart_flag = False

        self.step = 0

    def setEnvandSim(self, env, sim, ss, step, restart_flag):
        self.env = env
        self.sim = sim
        self.step = step
        self.temp = []
        self.ss = ss
        self.restart_flag = restart_flag

    def reset(self):
        self.sim.reset()
        self.ego = None
        self.sim.load(lgsvl.wise.DefaultAssets.map_borregasave)
        self.spawns = self.sim.get_spawn()
        state_npc = lgsvl.AgentState()
        state_npc.transform = self.spawns[0]
        state_npc.transform.position = Vector(165.646987915039, -4.49, -63)
        self.npc = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state_npc)
        state_npc.transform.position = Vector(165.646987915039, -4.49, -60)
        npc_1 = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state_npc)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        Vars = solution.variables
        print(Vars)
        # Vars=[2120, 30, 0.35, 8299, 800, 3000, 450, 39.4, 4, 1, 0.4, 0.8]
        x0 = float('%.3f' % Vars[0])
        x1 = float('%.3f' % Vars[1])
        x2 = float('%.3f' % Vars[2])
        x3 = float('%.3f' % Vars[3])
        x4 = float('%.3f' % Vars[4])
        x5 = float('%.3f' % Vars[5])
        x6 = float('%.3f' % Vars[6])
        x7 = float('%.3f' % Vars[7])
        x8 = float('%.3f' % Vars[8])
        x9 = float('%.3f' % Vars[9])
        x10 = float('%.3f' % Vars[10])
        x11 = float('%.3f' % Vars[11])

        str_temp = str(x0) + " " + str(x1) + " " + str(x2) + " " + str(x3) + " " + str(x4) + " " + str(x5) + " " + str(
            x6) + " " + str(x7) + " " + str(x8) + " " + str(
            x9) + " " + str(x10) + " " + str(x11) + " "

        change = []
        change_ratio = []

        # changes' precision
        d0 = float('%.3f' % (np.abs(x0 - 2120) / (2500 - 2000)))
        d1 = float('%.3f' % (np.abs(x1 - 30) / (60 - 20)))
        d2 = float('%.3f' % (np.abs(x2 - 0.35) / (0.39 - 0.30)))
        d3 = float('%.3f' % (np.abs(x3 - 8299) / (13000 - 6000)))
        d4 = float('%.3f' % (np.abs(x4 - 800) / (1100 - 600)))
        d5 = float('%.3f' % (np.abs(x5 - 3000) / (3150 - 2500)))
        d6 = float('%.3f' % (np.abs(x6 - 450) / (550 - 400)))
        d7 = float('%.3f' % (np.abs(x7 - 39.4) / (50 - 30)))
        d8 = float('%.3f' % (np.abs(x8 - 4) / (5 - 2)))
        d9 = float('%.3f' % (np.abs(x9 - 1) / (1.5 - 0.15)))
        d10 = float('%.3f' % (np.abs(x10 - 0.4) / (0.6 - 0.2)))
        d11 = float('%.3f' % (np.abs(x11 - 0.8) / (0.95 - 0.65)))

        # changes' ratio
        r0 = float('%.3f' % (np.abs(x0 - 2120) / 2120))
        r1 = float('%.3f' % (np.abs(x1 - 30) / 30))
        r2 = float('%.3f' % (np.abs(x2 - 0.35) / 0.35))
        r3 = float('%.3f' % (np.abs(x3 - 8299) / 8299))
        r4 = float('%.3f' % (np.abs(x4 - 800) / 800))
        r5 = float('%.3f' % (np.abs(x5 - 3000) / 3000))
        r6 = float('%.3f' % (np.abs(x6 - 450) / 450))
        r7 = float('%.3f' % (np.abs(x7 - 39.4) / 39.4))
        r8 = float('%.3f' % (np.abs(x8 - 4) / 4))
        r9 = float('%.3f' % (np.abs(x9 - 1) / 1))
        r10 = float('%.3f' % (np.abs(x10 - 0.4) / 0.4))
        r11 = float('%.3f' % (np.abs(x11 - 0.8) / 0.8))

        change_ratio.append(r0)
        change_ratio.append(r1)
        change_ratio.append(r2)
        change_ratio.append(r3)
        change_ratio.append(r4)
        change_ratio.append(r5)
        change_ratio.append(r6)
        change_ratio.append(r7)
        change_ratio.append(r8)
        change_ratio.append(r9)
        change_ratio.append(r10)
        change_ratio.append(r11)

        change_max = max(change_ratio)
        """
         greater than 1000  -> 0.01
         100 ~ 1000         -> 0.02
         1 ~ 100            -> 0.04
         0 ~ 1              -> 0.08
        """

        # mass
        if d0 < 0.02:

            x0 = 2120.0
        else:
            change.append(d0)

        # wheel_mass
        if d1 < 0.04:

            x1 = 30
        else:
            change.append(d1)

        # wheel_radius
        if d2 < 0.08:

            x2 = 0.35
        else:
            change.append(d2)

        # MaxRPM
        if d3 < 0.01:

            x3 = 8299
        else:
            change.append(d3)

        # MinRPM
        if d4 < 0.02:

            x4 = 800
        else:
            change.append(d4)

        # MaxBrakeTorque
        if d5 < 0.02:

            x5 = 3000
        else:
            change.append(d5)

        # MaxMotorTorque
        if d6 < 0.02:

            x6 = 450.0
        else:
            change.append(d6)

        # MaxSteeringAngle
        if d7 < 0.04:

            x7 = 39.4
        else:
            change.append(d7)

        # TireDragCoeff
        if d8 < 0.04:

            x8 = 4
        else:
            change.append(d8)

        # WheelDamping
        if d9 < 0.04:

            x9 = 1
        else:
            change.append(d9)

        # ShiftTime
        if d10 < 0.08:

            x10 = 0.4
        else:
            change.append(d10)

        # TractionControlSlipLimit
        if d11 < 0.08:

            x11 = 0.8
        else:
            change.append(d11)

        phy = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
        # phy = [2120, 30, 0.35, 8299, 800, 3000, 450, 39.4, 4, 1, 0.4, 0.8]
        # try:
        if self.step % 5 == 0 or self.restart_flag:
            if self.restart_flag:
                self.restart_flag = False
            # reset the envs
            try:
                self.reset()
                # set the new ego vehicle
                state = lgsvl.AgentState()
                state.transform = self.sim.map_point_on_lane(Vector(100.646987915039, -4.49, -45))
                self.ego = self.sim.add_agent(lgsvl.wise.DefaultAssets.ego_lincoln2017mkz_apollo5, lgsvl.AgentType.EGO,
                                              state)
                # to connect bridge
                self.ego.connect_bridge(
                    self.env.str("LGSVL__AUTOPILOT_0_HOST", lgsvl.wise.SimulatorSettings.bridge_host),
                    self.env.int("LGSVL__AUTOPILOT_0_PORT", lgsvl.wise.SimulatorSettings.bridge_port)
                )
            except Exception as e:
                self.env = None
                self.sim = None
                self.ego = None
                self.npc = None
                self.ss = None
                print("current step:", self.step)
                raise Exception(e)
            # to connect dv for the new ego vehicle
            dv = lgsvl.dreamview.Connection(self.sim, self.ego,
                                            self.env.str("LGSVL__AUTOPILOT_0_HOST", "192.168.50.51"))
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
            destination = self.spawns[0].destinations[0]
            print("ready to setup apollo")
            try:
                dv.setup_apollo(destination.position.x, destination.position.z, modules)
            except Exception as e:
                while not self.ego.bridge_connected:
                    time.sleep(1)
                try:
                    dv.reconnect()
                    dv.setup_apollo(destination.position.x, destination.position.z, modules)
                except Exception as e:
                    self.env = None
                    self.sim = None
                    self.ego = None
                    self.npc = None
                    self.ss = None
                    print("current step:", self.step)
                    raise Exception("dv time out,not bridge")
        else:
            # put the ego vehicle to the original location
            state = lgsvl.AgentState()
            state.transform = self.sim.map_point_on_lane(Vector(100.646987915039, -4.49, -45))
            self.ego.state = state
        try:
            result, TET, TIT, average_acc = lg.main(phy, self.env, self.sim, self.ego, self.npc, self.ss,self.step)
        except Exception as e:
            print(e)
            self.env = None
            self.sim = None
            self.ego = None
            self.npc = None
            self.ss = None
            f12 = float('%.3f' % change_max)
            f13 = float('%.3f' % 10.5)
            f14 = len(change)

            solution.objectives[0] = f12  # min maximum para change
            solution.objectives[1] = f13  # distance - speed
            solution.objectives[2] = f14  # changed para num
            # self.step = self.step + 1
            print("current step:", self.step)
            raise Exception("apollo or simulator comes up with problem")

        if result > 20:
            self.env = None
            self.sim = None
            self.ego = None
            self.npc = None
            self.ss = None
            f12 = float('%.3f' % change_max)
            f13 = float('%.3f' % result)
            f14 = len(change)

            solution.objectives[0] = f12  # min maximum para change
            solution.objectives[1] = 10.5  # distance - speed
            solution.objectives[2] = f14  # changed para num
            print("current step:", self.step)
            raise Exception("ego does not move!" + result)

        f12 = float('%.3f' % change_max)
        f13 = float('%.3f' % result)
        f14 = len(change)

        solution.objectives[0] = f12  # min maximum para change
        solution.objectives[1] = f13  # distance - speed
        solution.objectives[2] = f14  # changed para num
        self.step = self.step + 1
        print("step:", self.step)
        # ---------------------------------------------------------------------------------------------------------------
        # self.temp.append(
        #     str_temp + str(f12) + " " + str(f13) + " " + str(f14) + " " + str(TET) + " " + str(TIT) + " " + str(
        #         average_acc) + " ")
        # if self.step % 30 == 0:
        #     f_r_2 = open("file_storage.txt", 'a')
        #     print("**********write_temp****", len(self.temp), "****************")
        #     for temp in self.temp:
        #         f_r_2.write(temp)
        #         f_r_2.write("\n")
        #     self.temp = []
        #     f_r_2.close()
        # # ---------------------------------------------------------------------------------------------------------------
        temp = str_temp + str(f12) + " " + str(f13) + " " + str(f14) + " " + str(TET) + " " + str(TIT) + " " + str(
                average_acc) + " "
        f_r_2 = open("file_storage.txt", 'a')
        f_r_2.write(temp)
        f_r_2.write("\n")
        f_r_2.close()


        return solution

    def get_name(self):
        return 'LGSVLProblem'
