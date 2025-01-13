import os
import csv
import signal
import time
from datetime import datetime

import ray
from copy import deepcopy
from leaderboard.nocrash_evaluator import NoCrashEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager
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
from jmetal.util.observer import Observer

class NoCrashEvalRunner():
    def __init__(self, args, debug=False):
        args = deepcopy(args)
        # args.port = port
        # args.trafficManagerPort = tm_port
        args.debug = debug
        args.record = ''
        self.runner = NoCrashEvaluator(args, StatisticsManager())
        # self.runner = NoCrashEvaluator(args, StatisticsManager(args))
        self.args = args
        self.problem = CarlaProblem(self.args, self.runner, is_rerun = False)




    def rerun(self):
        pickle_file = open('myalgorithm.pkl', 'rb')
        runner = pickle.load(pickle_file)
        pickle_file.close()
        self.all_sol = runner.all_sol
        self.all_config = runner.all_config
        self.all_comb = runner.all_comb

        self.cur_eva = runner.cur_eva
        self.flag = runner.flag
        self.stop_signal = runner.stop_signal

        self.next_range = runner.next_range
        self.next_max_nums = runner.next_max_nums

        self.cur_eva = self.pop_size * (int)(self.cur_eva / self.pop_size)
        self.all_sol = self.all_sol[:self.cur_eva]
        self.all_config = self.all_config[:self.cur_eva]
        self.all_comb = self.all_comb[:self.cur_eva]

        self.cur_5_sol = runner.cur_5_sol
        self.cur_5_phy_config = runner.cur_5_phy_config

        index = self.pop_size * (int)(len(self.cur_5_sol) / self.pop_size)
        self.cur_5_sol=self.cur_5_sol[:index]
        self.cur_5_phy_config = self.cur_5_phy_config[:index]
        print(len(self.all_comb))
        print(len(self.cur_5_phy_config))
        print(len(self.cur_5_sol))

    def run(self,phy0):
        # phy0 = [5800.000, 0.150, 2.000, 0.350, 0.500, 10.000, 2404.00, 0.300, 3.500, 0.150, 35.500, 1500.000]

        # phy0 =[]
        # try:
        #     # self.rerun()
        #     time.sleep(2)
        #     result, TET, TIT, acc = self.runner.run(self.args, phy0)
        #     # temp_run = NoCrashEvaluator(self.args, StatisticsManager())
        #     # result, TET, TIT, acc = temp_run.run(self.args, phy0)
        #     # del temp_run
        #     # self.algothrim_run()
        # except Exception as e:
        #     print("catch error:", e.args)
        #     self.runner = None
        #     self.problem = None
        #     save_file = open('myalgorithm.pkl', 'wb')
        #     pickle.dump(self, save_file)
        #     save_file.close()
        #     print("save pkl")
        #     print("cur_eva :", self.cur_eva)
        result, TET, TIT, acc = self.runner.run(self.args, phy0)
        return result, TET, TIT, acc
    def search(self, rerun = False):
        algorithm = NSGAII(
            problem=self.problem,
            population_size=50,
            offspring_population_size=50,
            mutation=PolynomialMutation(probability=1.0 / self.problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=5000)
        )
        """
            reload pickle
        """
        #
        # pickle_file  = open('/home/new_drive2/mjw/mmfn/outputs/search_pkl/nsga.pkl', 'rb')
        # al = pickle.load(pickle_file)
        # pickle_file.close()
        # algorithm = al
        #
        # algorithm.runner = self.runner
        # algorithm.problem = self.problem
        #
        #
        # eva = None
        # eva = algorithm.get_eva()
        # print("current eva:", eva)
        # print(algorithm.evaluations)
        # print(al.evaluations)
        # algorithm.rerun()
        '''
            no pickle
        '''
        try:
            # print("abc")
            # custom_observer = CustomObserver()

            if (not rerun):
                # algorithm.observable.register(custom_observer)
                # print(algorithm)
                algorithm.run()
            # raise Exception()
            else:
                pickle_file = open('/home/simplexity/mjw/mmfn/outputs/search_pkl/nsga.pkl', 'rb')
                al = pickle.load(pickle_file)
                pickle_file.close()
                algorithm = al
                # algorithm.observable.register(custom_observer)
                with open("/home/simplexity/mjw/mmfn/outputs/output_storage/file_storage.txt", 'r') as file:
                    lines = file.readlines()
                total_lines = len(lines)
                lines_to_keep = total_lines - (total_lines % algorithm.population_size)

                trimmed_lines = lines[:lines_to_keep]

                with open("/home/simplexity/mjw/mmfn/outputs/output_storage/file_storage.txt", 'w') as file:
                    file.writelines(trimmed_lines)

                algorithm.runner = self.runner
                self.problem.is_rerun = True
                algorithm.problem = self.problem

                eva = None
                eva = algorithm.get_eva()
                print("current eva:", eva)
                print(algorithm.evaluations)
                print(al.evaluations)
                algorithm.rerun()

                front = get_non_dominated_solutions(algorithm.get_result())

                # Save results to file
                print_function_values_to_file(front, 'FUN.' + algorithm.label)
                print_variables_to_file(front, 'VAR.' + algorithm.label)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_file_name = f"/home/simplexity/mjw/mmfn/outputs/output_storage/file_storage_final_{timestamp}.txt"
                os.rename("/home/simplexity/mjw/mmfn/outputs/output_storage/file_storage.txt", new_file_name)

        except Exception as e:
            algorithm.runner = None
            algorithm.problem = None
            print("catch error:", e.args)
            save_file = open('/home/simplexity/mjw/mmfn/outputs/search_pkl/nsga.pkl', 'wb')
            pickle.dump(algorithm, save_file)
            save_file.close()
            print("SAVE pkl")
        finally:
            print("done!")

                # print(f'Algorithm: ${algorithm.get_name()}')
        # print(f'Problem: ${self.problem.get_name()}')
        # print(f'Computing time: ${algorithm.total_computing_time}')
        #
        # plot_front = Plot(title='Pareto Front',
        #                   axis_labels=['min maximum parameter change', 'min distance - speed', 'min changed para num'])
        # plot_front.plot(front, label='Three Objectives', filename='Pareto rain Front', format='png')
# class CustomObserver(Observer):
#     def __init__(self,frequency = 1):
#         self.frequency = frequency
#         self.counter = 0
#     def update(self, *args, **kwargs):
#         self.counter +=1
#         print(kwargs)
#         if self.counter%self.frequency == 0:
#             algo = kwargs.get('algorithm', None)
#             algo.runner = None
#             algo.problem = None
#             # print("catch error:", e.args)
#             print("SAVE_PKL")
#             save_file = open('/home/new_drive2/mjw/mmfn/outputs/search_pkl/nsga.pkl', 'wb')
#             pickle.dump(algo, save_file)
#             save_file.close()
#             algo.runner = NoCrashEvaluator(algo.args, StatisticsManager())
#             algo.problem = CarlaProblem(algo.args, algo.runner)

class CarlaProblem(FloatProblem):

    def __init__(self, args, runner, is_rerun):
        super(CarlaProblem, self).__init__()
        self.number_of_variables = 12
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.args = args
        self.runner = runner

        self.obj_direction = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(1)', 'f(2)', 'f(3)']

        self.lower_bound = [4200, 0.1, 1.0, 0.2, 0.3, 8.0, 1940, 0.2, 1.0, 0.2, 31.7, 1200]
        self.upper_bound = [5900, 0.2, 3.0, 0.4, 0.6, 12.0, 2700, 0.5, 3.0, 0.3, 36.0, 1600]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        self.run_time = 0
        self.error_time = 0

        self.is_rerun = is_rerun

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        Vars = solution.variables
        # print(Vars)

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
        f_r = open("/home/simplexity/mjw/mmfn/outputs/output_storage/file_storage.txt", 'a')

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
        d0 = float('%.3f' % (np.abs(x0 - 5800) / (5900 - 4200)))
        d1 = float('%.3f' % (np.abs(x1 - 0.15) / (0.2 - 0.1)))
        d2 = float('%.3f' % (np.abs(x2 - 2) / (3 - 1)))
        d3 = float('%.3f' % (np.abs(x3 - 0.35) / (0.4 - 0.2)))
        d4 = float('%.3f' % (np.abs(x4 - 0.5) / (0.6 - 0.3)))
        d5 = float('%.3f' % (np.abs(x5 - 10) / (12 - 8)))
        d6 = float('%.3f' % (np.abs(x6 - 2404) / (2700 - 1940)))
        d7 = float('%.3f' % (np.abs(x7 - 0.3) / (0.5 - 0.2)))
        d8 = float('%.3f' % (np.abs(x8 - 2) / (3 - 1)))
        d9 = float('%.3f' % (np.abs(x9 - 0.25) / (0.3 - 0.2)))
        d10 = float('%.3f' % (np.abs(x10 - 31.7) / (36.0 - 31.7)))
        d11 = float('%.3f' % (np.abs(x11 - 1500) / (1600 - 1200)))

        # changes' ratio
        r0 = float('%.3f' % (np.abs(x0 - 5800) / 5800))
        r1 = float('%.3f' % (np.abs(x1 - 0.15) / 0.15))
        r2 = float('%.3f' % (np.abs(x2 - 2) / 2))
        r3 = float('%.3f' % (np.abs(x3 - 0.35) / 0.35))
        r4 = float('%.3f' % (np.abs(x4 - 0.5) / 0.5))
        r5 = float('%.3f' % (np.abs(x5 - 10) / 10))
        r6 = float('%.3f' % (np.abs(x6 - 2404) / 2404))
        r7 = float('%.3f' % (np.abs(x7 - 0.3) / 0.3))
        r8 = float('%.3f' % (np.abs(x8 - 2) / 2))
        r9 = float('%.3f' % (np.abs(x9 - 0.25) / 0.25))
        r10 = float('%.3f' % (np.abs(x10 - 31.7) / 31.7))
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

            x8 = 2.0
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

        # result = self.runner.run(self.args, physics)
        # result = self.run(physics)
        self.run_time = self.run_time + 1
        # if(self.run_time %1702 ==0 and self.is_rerun == False): #1652
        if (self.run_time % 1702 == 0 ):  # 1652
            raise Exception()
        try:
            result, TET, TIT, acc = self.timed_run(self.runner, self.args, physics)
            # result, TET, TIT, acc = self.runner.run(self.args,physics)
        except:
            raise Exception()
        # while True:
        #     # time.sleep(2)
        #     result, TET, TIT, acc = NoCrashEvalRunner(self.args).run(physics)
        #     if round(result, 3) != 11.549 or TIT != 0.0 or TET != 0.0:
        #         break
        #     retry_time+=1
        #     if retry_time%2==0:
        #         if retry_time%6 == 0:
        #             temp_run = NoCrashEvaluator(self.args, StatisticsManager())
        #             result, TET, TIT, acc = temp_run.run(self.args,physics)
        #             del temp_run
        #             break
        #         # else:
        #         #     del self.runner
        #         #     self.runner = NoCrashEvalRunner(self.args)
        #         print("RENEW RUNNER")
        # print(self.run_time)

        # print(result)

        print(self.run_time, end="\t")
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
        f_r.write("\n")

        solution.objectives[0] = f12  # min maximum para change
        solution.objectives[1] = f13  # distance - speed
        solution.objectives[2] = f14  # changed para num

        return solution

    def run(self, physics):

        # town = self.args.town
        # weather = self.args.weather
        #
        # port = self.args.port
        # tm_port = port + 2
        # runner = NoCrashEvalRunner(self.args, town, weather, port=port, tm_port=tm_port)
        # result = runner.run(physics)
        # del runner
        # runner = NoCrashEvalRunner(self.args)
        # if (self.run_time % 2 == 0):
        #     del self.runner
        #     self.runner = NoCrashEvalRunner(self.args)

        # while True:
        #     time.sleep(2)
        #     result, TET, TIT, acc = self.runner.run(physics)
        #     if round(result,3) != 11.549 or TIT !=0.0 or TET != 0.0:
        #         break
        # result, TET, TIT, acc = self.runner.run(physics)
        result, TET, TIT, acc = self.timed_run(self.runner, physics)
        # del runner
        # gc.collect()
        # print(result)
        # time.sleep(3)
        return result

    def get_name(self):
        return 'CarlaProblem'

    def timed_run(self,fun, args,physics,timeout=25):
        try:
            signal.signal(signal.SIGALRM, self.timeoutHandler)
            signal.alarm(timeout)

            result, TET, TIT, acc = fun.run(args,physics)
            signal.alarm(0)
            return result, TET, TIT, acc
        except Exception as e:
            print(f"ERROR:might be time out")

        raise(Exception)
    def timeoutHandler(self,signum,frame):
        raise Exception
