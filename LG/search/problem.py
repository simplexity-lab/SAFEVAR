import random
import numpy as np
# import simplexity_carla_combination as cc
from testing_LG import LG
from jmetal.core.problem import FloatProblem, BinaryProblem, Problem
from jmetal.core.solution import FloatSolution, BinarySolution, IntegerSolution, CompositeSolution


class CarlaProblem(FloatProblem):

    def __init__(self):
        super(CarlaProblem, self).__init__()
        self.number_of_variables = 12
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_direction = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(1)', 'f(2)', 'f(3)']

        self.lower_bound = [2000, 20, 0.30, 6000, 600, 2500, 400, 30, 2, 0.15, 0.2, 0.65]
        self.upper_bound = [2500, 60, 0.39, 13000, 1100, 3150, 550, 50, 5, 1.5, 0.6, 0.95]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        self.lg = LG()

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        Vars = solution.variables
        print(Vars)

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
        r1 = float('%.3f' % (np.abs(x1 - 0.30) / 30))
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

        # result = cc.main(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)
        phy = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
        self.lg.create_ego_and_connect()
        self.lg.set_npc_follow()
        self.lg.set_physics(phy)
        result = self.lg.run()
        self.lg.destory()
        print(result)


        # f0 = float('%.3f' % (np.abs(x0 - 5000) / (6400 - 4200)))
        # f1 = float('%.3f' % (np.abs(x1 - 0.15) / (0.2 - 0.1)))
        # f2 = float('%.3f' % (np.abs(x2 - 2) / (3.0 - 1.0)))
        # f3 = float('%.3f' % (np.abs(x3 - 0.35) / (0.4 - 0.2)))
        # f4 = float('%.3f' % (np.abs(x4 - 0.5) / (0.6 - 0.3)))
        # f5 = float('%.3f' % (np.abs(x5 - 10) / (12.0 - 8.0)))
        # f6 = float('%.3f' % (np.abs(x6 - 1340) / (2100 - 1340)))
        # f7 = float('%.3f' % (np.abs(x7 - 0.3) / (0.5 - 0.2)))
        # f8 = float('%.3f' % (np.abs(x8 - 2) / (3.0 - 1.0)))
        # f9 = float('%.3f' % (np.abs(x9 - 0.25) / (0.3 - 0.2)))
        # f10 = float('%.3f' % (np.abs(x10 - 31.7) / (35.6 - 31.7)))
        # f11 = float('%.3f' % (np.abs(x11 - 1500) / (1800 - 1200)))

        f12 = float('%.3f' % change_max)
        f13 = float('%.3f' % result)
        f14 = len(change)

        solution.objectives[0] = f12  # min maximum para change
        solution.objectives[1] = f13  # distance - speed
        solution.objectives[2] = f14  # changed para num

        return solution

    def get_name(self):
        return 'CarlaProblem'
