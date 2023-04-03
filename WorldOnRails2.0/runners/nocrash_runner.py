import os
import csv
import ray
from copy import deepcopy
from leaderboard.nocrash_evaluator import NoCrashEvaluator

import random
import numpy as np

from jmetal.core.problem import FloatProblem, BinaryProblem, Problem
from jmetal.core.solution import FloatSolution, BinarySolution, IntegerSolution, CompositeSolution
# ---------------------------------------------------------------------------------------------------
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.algorithm.multiobjective.random_search import RandomSearch
from jmetal.lab.visualization import Plot
import dill as pickle


class NoCrashEvalRunner():
    def __init__(self, args, town, weather, port=1000, tm_port=1002, debug=False):
        args = deepcopy(args)

        # Inject args
        args.scenario_class = 'nocrash_eval_scenario'
        args.port = port
        args.trafficManagerPort = tm_port
        args.debug = debug
        args.record = ''

        args.town = town
        args.weather = weather


        self.runner = NoCrashEvaluator(args, StatisticsManager(args))
        self.args = args
        self.problem = CarlaProblem(self.runner, self.args)

    def run(self):
        phy0 = [5800.000, 0.150, 2.000, 0.350, 0.500, 10.000, 2404.00, 0.300, 3.500, 0.150, 35.500, 1500.000]

        # return self.runner.run(self.args, phy0)
        if self.args.OrigiCon == 0:
            return self.runner.run(self.args, phy0)
        else:
            return self.search()
        # return  self.RS_search()

    def search(self):
        algorithm = NSGAII(
            problem=self.problem,
            population_size=50,
            offspring_population_size=50,
            mutation=PolynomialMutation(probability=1.0 / self.problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=5000)
        )
        try:
            algorithm.run()
            # algorithm.rerun()
        except Exception as e:
            print("catch error:", e.args)
        finally:
            print("done!")

        front = get_non_dominated_solutions(algorithm.get_result())

        # Save results to file
        print_function_values_to_file(front, 'FUN.' + algorithm.label)
        print_variables_to_file(front, 'VAR.' + algorithm.label)

        print(f'Algorithm: ${algorithm.get_name()}')
        print(f'Problem: ${self.problem.get_name()}')
        print(f'Computing time: ${algorithm.total_computing_time}')

        plot_front = Plot(title='Pareto Front',
                          axis_labels=['min maximum parameter change', 'min distance - speed', 'min changed para num'])
        plot_front.plot(front, label='Three Objectives', filename='Pareto rain Front', format='png')

    def RS_search(self):
        max_evaluations = 5000
        #
        algorithm = RandomSearch(
            problem=self.problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )

        algorithm.run()

        try:
            algorithm.run()
        except Exception as e:
            print("catch error:", e.args)
        finally:
            print("done!")
        front = algorithm.get_result()

        from jmetal.lab.visualization import Plot

        # Save results to file
        print_function_values_to_file(front, 'FUN.' + algorithm.label)
        print_variables_to_file(front, 'VAR.' + algorithm.label)

        print(f'Algorithm: ${algorithm.get_name()}')
        print(f'Problem: ${self.problem.get_name()}')
        print(f'Computing time: ${algorithm.total_computing_time}')

        plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
        plot_front.plot(front, label='Three Objectives', filename='RS Pareto Front', format='png')


# ---------------------------------------------------------------------------------
class CarlaProblem(FloatProblem):

    def __init__(self, runner, args):
        super(CarlaProblem, self).__init__()
        self.number_of_variables = 12
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.runner = runner
        self.args = args
        self.count = 0

        self.obj_direction = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(1)', 'f(2)', 'f(3)']
        # phy = [5800, 0.15, 2.0, 0.35, 0.5, 10.0, 2404, 0.3, 3.5, 0.25, 35.5, 1500]
        self.lower_bound = [4200, 0.1, 1.0, 0.2, 0.3, 8.0, 2040, 0.2, 1.0, 0.2, 31.7, 1200]
        self.upper_bound = [7000, 0.2, 3.0, 0.4, 0.6, 12.0, 2700, 0.5, 3.9, 0.3, 37.0, 1650]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def set_runner(self, runner):
        self.runner = runner

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

        global f_r
        f_r = open("file_storage.txt", 'a')

        f_r.write(str(x0))
        f_r.write(' ')
        f_r.write(str(x1))
        f_r.write(' ')
        f_r.write(str(x2))
        f_r.write(' ')
        f_r.write(str(x3))
        f_r.write(' ')
        f_r.write(str(x4))
        f_r.write(' ')
        f_r.write(str(x5))
        f_r.write(' ')
        f_r.write(str(x6))
        f_r.write(' ')
        f_r.write(str(x7))
        f_r.write(' ')
        f_r.write(str(x8))
        f_r.write(' ')
        f_r.write(str(x9))
        f_r.write(' ')
        f_r.write(str(x10))
        f_r.write(' ')
        f_r.write(str(x11))
        f_r.write(' ')

        change = []
        change_ratio = []

        # changes' precision
        d0 = float('%.3f' % (np.abs(x0 - 5800) / (7000 - 4200)))
        d1 = float('%.3f' % (np.abs(x1 - 0.15) / (0.2 - 0.1)))
        d2 = float('%.3f' % (np.abs(x2 - 2) / (3 - 1)))
        d3 = float('%.3f' % (np.abs(x3 - 0.35) / (0.4 - 0.2)))
        d4 = float('%.3f' % (np.abs(x4 - 0.5) / (0.6 - 0.3)))
        d5 = float('%.3f' % (np.abs(x5 - 10) / (12 - 8)))
        d6 = float('%.3f' % (np.abs(x6 - 2404) / (2700 - 2040)))
        d7 = float('%.3f' % (np.abs(x7 - 0.3) / (0.5 - 0.2)))
        d8 = float('%.3f' % (np.abs(x8 - 3.5) / (3.9 - 1)))
        d9 = float('%.3f' % (np.abs(x9 - 0.25) / (0.3 - 0.2)))
        d10 = float('%.3f' % (np.abs(x10 - 35.5) / (37.0 - 31.7)))
        d11 = float('%.3f' % (np.abs(x11 - 1500) / (1650 - 1200)))

        # changes' ratio
        r0 = float('%.3f' % (np.abs(x0 - 5800) / 5800))
        r1 = float('%.3f' % (np.abs(x1 - 0.15) / 0.15))
        r2 = float('%.3f' % (np.abs(x2 - 2) / 2))
        r3 = float('%.3f' % (np.abs(x3 - 0.35) / 0.35))
        r4 = float('%.3f' % (np.abs(x4 - 0.5) / 0.5))
        r5 = float('%.3f' % (np.abs(x5 - 10) / 10))
        r6 = float('%.3f' % (np.abs(x6 - 2404) / 2404))
        r7 = float('%.3f' % (np.abs(x7 - 0.3) / 0.3))
        r8 = float('%.3f' % (np.abs(x8 - 3.5) / 3.5))
        r9 = float('%.3f' % (np.abs(x9 - 0.25) / 0.25))
        r10 = float('%.3f' % (np.abs(x10 - 35.5) / 35.5))
        r11 = float('%.3f' % (np.abs(x11 - 1500) / 1500))

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

        # max_rpm
        if d0 < 0.01:

            x0 = 5800.0
        else:
            change.append(d0)

        # damping_rate_full_throttle
        if d1 < 0.08:

            x1 = 0.15
        else:
            change.append(d1)

        # damping_rate_zero_throttle_clutch_engaged
        if d2 < 0.04:

            x2 = 2.0
        else:
            change.append(d2)

        # damping_rate_zero_throttle_clutch_disengaged
        if d3 < 0.08:

            x3 = 0.35
        else:
            change.append(d3)

        # gear_switch_time
        if d4 < 0.08:

            x4 = 0.5
        else:
            change.append(d4)

        # clutch_strength
        if d5 < 0.04:

            x5 = 10.0
        else:
            change.append(d5)

        # mass
        if d6 < 0.02:

            x6 = 2404.0
        else:
            change.append(d6)

        # drag_coefficient
        if d7 < 0.08:

            x7 = 0.3
        else:
            change.append(d7)

        # tire_friction
        if d8 < 0.04:

            x8 = 3.5
        else:
            change.append(d8)

        # damping_rate
        if d9 < 0.08:

            x9 = 0.25
        else:
            change.append(d9)

        # radius
        if d10 < 0.04:

            x10 = 35.5
        else:
            change.append(d10)

        # max_brake_torque
        if d11 < 0.02:

            x11 = 1500.0
        else:
            change.append(d11)

        physics = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
        if len(change) == 0:
            physics = []
        try:
            result, TET, TIT, acc = self.runner.run(self.args, physics)
        except Exception as e:
            print("error:", e.args)
            self.runner = None
            f12 = float('%.3f' % change_max)
            f13 = float('%.3f' % 6.0)
            f14 = len(change)
            f_r.write(str(f12))
            f_r.write(' ')
            f_r.write(str(f13))
            f_r.write(' ')
            f_r.write(str(f14))
            f_r.write(' ')
            f_r.write(str(TET))
            f_r.write(' ')
            f_r.write(str(TIT))
            f_r.write(' ')
            f_r.write(str(acc))
            f_r.write(' ')
            f_r.write("\n")
            f_r.close()
            solution.objectives[0] = f12  # min maximum para change
            solution.objectives[1] = f13  # distance - speed
            solution.objectives[2] = f14  # changed para num
            raise Exception(e)

        # print(result)

        f12 = float('%.3f' % change_max)
        f13 = float('%.3f' % result)
        f14 = len(change)

        f_r.write(str(f12))
        f_r.write(' ')
        f_r.write(str(f13))
        f_r.write(' ')
        f_r.write(str(f14))
        f_r.write(' ')
        f_r.write(str(TET))
        f_r.write(' ')
        f_r.write(str(TIT))
        f_r.write(' ')
        f_r.write(str(acc))
        f_r.write(' ')
        f_r.write("\n")
        f_r.close()
        solution.objectives[0] = f12  # min maximum para change
        solution.objectives[1] = f13  # distance - speed
        solution.objectives[2] = f14  # changed para num

        self.count += 1
        return solution

    def get_name(self):
        return 'CarlaProblem'


# ---------------------------------------------------------------------------------

class StatisticsManager:
    headers = [
        'town',
        'traffic',
        'weather',
        'start',
        'target',
        'route_completion',
        'lights_ran',
        'duration',
        'distance',
    ]

    def __init__(self, args):

        self.finished_tasks = {
            'Town01': {},
            'Town02': {}
        }

        logdir = args.agent_config.replace('.yaml', '.csv')

        if args.resume and os.path.exists(logdir):
            self.load(logdir)
            self.csv_file = open(logdir, 'a')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
        else:
            self.csv_file = open(logdir, 'w')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
            self.csv_writer.writeheader()

    def load(self, logdir):
        with open(logdir, 'r') as file:
            log = csv.DictReader(file)
            for row in log:
                self.finished_tasks[row['town']][(
                    int(row['traffic']),
                    int(row['weather']),
                    int(row['start']),
                    int(row['target']),
                )] = [
                    float(row['route_completion']),
                    int(row['lights_ran']),
                    float(row['duration']),
                    float(row['distance']),
                ]

    def log(self, town, traffic, weather, start, target, route_completion, lights_ran, duration, distance):
        self.csv_writer.writerow({
            'town': town,
            'traffic': traffic,
            'weather': weather,
            'start': start,
            'target': target,
            'route_completion': route_completion,
            'lights_ran': lights_ran,
            'duration': duration,
            'distance': distance,
        })

        self.csv_file.flush()

    def is_finished(self, town, route, weather, traffic):
        start, target = route
        key = (int(traffic), int(weather), int(start), int(target))
        return key in self.finished_tasks[town]
