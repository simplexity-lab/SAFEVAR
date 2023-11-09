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

        # ---------------------------------------------------------------------------------------------------------------
        self.pop_size = 50

        self.next_range = [[4200, 7000], [0.1, 0.2], [1, 3], [0.2, 0.4], [0.3, 0.6], [8, 12], [2040, 2700], [0.2, 0.5],
                           [1, 3.9],
                           [0.2, 0.3], [31.7, 37], [1200, 1650]]
        self.next_max_nums = 12
        self.or_sd = 2.2
        self.or_value = [5800, 0.15, 2.0, 0.35, 0.5, 10.0, 2404, 0.3, 3.5, 0.25, 35.5, 1500]

        # sol 50*3  config 50*12
        self.all_sol = []
        self.all_config = []
        self.all_comb = []

        self.cur_eva = 0
        self.max_evaluation = 5000
        self.flag = True
        self.stop_signal = 0
        self.cur_5_phy_config = []
        self.cur_5_sol = []

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

    def run(self):
        # phy0 = [5800.000, 0.150, 2.000, 0.350, 0.500, 10.000, 2404.00, 0.300, 3.500, 0.150, 35.500, 1500.000]

        # phy0 =[]
        try:
            # self.rerun()
            # self.search()
            # self.RS_search()
            self.algothrim_run()
        except Exception as e:
            print("catch error:", e.args)
            self.runner = None
            self.problem = None
            save_file = open('myalgorithm.pkl', 'wb')
            pickle.dump(self, save_file)
            save_file.close()
            print("save pkl")
            print("cur_eva :", self.cur_eva)
        return
        # phy0 =[4768.512945499018, 0.1697922243522476, 1.899636669708449, 0.3062621346074821, 0.5895775696561507, 10.713798809082892, 2580.342123796868, 0.33255521319972803, 1.115415024099688, 0.2771708649219944, 32.0103699194326, 1632.2832068388889]
        # return self.runner.run(self.args, phy0)
        #
        # return self.search()
        # return  self.RS_search()

    def algothrim_run(self):
        while self.cur_eva < self.max_evaluation and self.flag == True:
            cur_gen_config = self.random_gen(self.pop_size, self.next_range, self.next_max_nums)
            cur_gen_sol = []
            cur_gen_resl_phy = []
            for i in range(0, self.pop_size):
                sol = []
                # 过滤一下 para是否大于最小改变阈值
                x0 = float('%.3f' % cur_gen_config[i][0])
                x1 = float('%.3f' % cur_gen_config[i][1])
                x2 = float('%.3f' % cur_gen_config[i][2])
                x3 = float('%.3f' % cur_gen_config[i][3])
                x4 = float('%.3f' % cur_gen_config[i][4])
                x5 = float('%.3f' % cur_gen_config[i][5])
                x6 = float('%.3f' % cur_gen_config[i][6])
                x7 = float('%.3f' % cur_gen_config[i][7])
                x8 = float('%.3f' % cur_gen_config[i][8])
                x9 = float('%.3f' % cur_gen_config[i][9])
                x10 = float('%.3f' % cur_gen_config[i][10])
                x11 = float('%.3f' % cur_gen_config[i][11])

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
                #
                physics = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
                if len(change) == 0:
                    physics = [5800.000, 0.150, 2.000, 0.350, 0.500, 10.000, 2404.00, 0.300, 3.500, 0.150, 35.500,
                               1500.000]
                try:

                    SD, tet, tit, avgDece = self.runner.run(self.args, physics)
                except Exception as e:
                    print("error:", e.args)
                    self.runner = None
                    self.cur_eva = self.pop_size * (int)(self.cur_eva / self.pop_size)
                    self.all_sol = self.all_sol[:self.cur_eva]
                    self.all_config = self.all_config[:self.cur_eva]
                    self.all_comb = self.all_comb[:self.cur_eva]
                    f12 = float('%.3f' % change_max)
                    f13 = float('%.3f' % 6.0)
                    f14 = len(change)
                    f_r.write(str(f12))
                    f_r.write(' ')
                    f_r.write(str(f13))
                    f_r.write(' ')
                    f_r.write(str(f14))
                    f_r.write(' ')
                    f_r.write(str(tet))
                    f_r.write(' ')
                    f_r.write(str(tit))
                    f_r.write(' ')
                    f_r.write(str(avgDece))
                    f_r.write(' ')
                    f_r.write("\n")
                    f_r.close()
                    raise Exception(e)
                MaxPC = float('%.3f' % change_max)
                SD = float('%.3f' % SD)
                MaxNums = len(change)
                # 存到汇总数组里面
                sol.append(MaxPC)
                sol.append(SD)
                sol.append(MaxNums)
                tmp = np.concatenate((np.array(cur_gen_config[i]), np.array(sol)), axis=0)
                # 收集所有的解
                self.all_comb.append(tmp.tolist())
                self.all_sol.append(sol)
                self.all_config.append(cur_gen_config[i])
                self.cur_eva += 1
                print("cur_eva:", self.cur_eva)
                cur_gen_sol.append(sol)
                # 收集每个参数真实投入的config 去计算下一次
                cur_gen_resl_phy.append(physics)
                self.cur_5_phy_config.append(physics)
                self.cur_5_sol.append(sol)
                # write to file
                f_r.write(str(MaxPC))
                f_r.write(' ')
                f_r.write(str(SD))
                f_r.write(' ')
                f_r.write(str(MaxNums))
                f_r.write(' ')
                f_r.write(str(tet))
                f_r.write(' ')
                f_r.write(str(tit))
                f_r.write(' ')
                f_r.write(str(avgDece))
                f_r.write(' ')
                f_r.write("\n")
                f_r.close()
            # 二维拼接
            # cur_gen_all = np.concatenate((np.array(cur_gen_config),np.array(cur_gen_sol)),axis=1)
            # 500 代过后 每代都检查一下收敛，连续5次收敛 停止循环
            isConverge = False
            if self.cur_eva > 500:
                isConverge = self.is_converge(self.all_comb)
                if isConverge:
                    self.stop_signal += 1
                else:
                    self.stop_signal = 0
                if self.stop_signal == 5:
                    self.flag = False
            if self.cur_eva >= 500 and self.cur_eva % (self.pop_size * 5) == 0:
                self.next_max_nums, self.next_range = self.cal_next_range_maxNums(self.cur_5_phy_config, self.cur_5_sol,
                                                                                  self.or_sd,
                                                                                  self.or_value,
                                                                                  self.next_range, self.next_max_nums)
                self.cur_5_phy_config = []
                self.cur_5_sol = []
            global f_e
            f_e = open("gen_next_nums_range.txt", 'a')
            f_e.write(str(self.next_max_nums))
            f_e.write(":")
            for i in self.next_range:
                f_e.write(str(i))
            f_e.write("::stop_signal:")
            f_e.write(str(self.stop_signal))
            f_e.write("::isConverage::")
            f_e.write(str(isConverge))
            f_e.write("\n")
            f_e.close()
            print("cur_gen:", self.cur_eva / self.pop_size)
            # if self.cur_eva == 1500:
            #     # save all_config,all_
            #     self.runner = None
            #     print("time to end")
            #     raise Exception("over")

        final_comb = self.cal_final_best(self.all_comb)
        np.savetxt('final_comb.txt', final_comb, fmt="%.3f", delimiter=" ")

    def random_gen(self, pop_size, rangeRefer, maxChangedNum):
        # range 是一个二维的数组 12 * 2 12个参数的上下届
        # res_params : pop_size * 12 的二维数组； 初始都是各自的初始值。根据select_arr选择改变的参数;
        res_params = [[5800, 0.15, 2.0, 0.35, 0.5, 10.0, 2404, 0.3, 3.5, 0.25, 35.5, 1500] for _ in range(0, pop_size)]
        for i in range(0, pop_size):
            res_changedNum = random.randint(1, maxChangedNum)
            # print(res_changedNum)
            selected_arr = random.sample(range(0, 12), res_changedNum)
            # record = []
            for j in range(0, len(selected_arr)):
                x = random.uniform(rangeRefer[selected_arr[j]][0],
                                   rangeRefer[selected_arr[j]][1])
                while self.isChange(selected_arr[j], x) is False:
                    x = random.uniform(rangeRefer[selected_arr[j]][0],
                                       rangeRefer[selected_arr[j]][1])

                res_params[i][selected_arr[j]] = x

        return res_params

    def isChange(self, index, value):
        value = float('%.3f' % value)
        if index == 0:
            d0 = float('%.3f' % (np.abs(value - 5800) / (7000 - 4200)))
            if d0 < 0.01:

                return False
            else:
                return True

        elif index == 1:
            d1 = float('%.3f' % (np.abs(value - 0.15) / (0.2 - 0.1)))
            if d1 < 0.08:

                return False
            else:
                return True
        elif index == 2:
            d2 = float('%.3f' % (np.abs(value - 2) / (3 - 1)))
            if d2 < 0.04:

                return False
            else:
                return True
        elif index == 3:
            d3 = float('%.3f' % (np.abs(value - 0.35) / (0.4 - 0.2)))
            if d3 < 0.08:

                return False
            else:
                return True

        elif index == 4:
            d4 = float('%.3f' % (np.abs(value - 0.5) / (0.6 - 0.3)))
            if d4 < 0.08:

                return False
            else:
                return True

        elif index == 5:
            d5 = float('%.3f' % (np.abs(value - 10) / (12 - 8)))
            if d5 < 0.04:

                return False
            else:
                return True
        elif index == 6:
            d6 = float('%.3f' % (np.abs(value - 2404) / (2700 - 2040)))
            if d6 < 0.02:

                return False
            else:
                return True
        elif index == 7:
            d7 = float('%.3f' % (np.abs(value - 0.3) / (0.5 - 0.2)))
            if d7 < 0.08:

                return False
            else:
                return True

        elif index == 8:
            d8 = float('%.3f' % (np.abs(value - 3.5) / (3.9 - 1)))
            if d8 < 0.04:

                return False
            else:
                return True

        elif index == 9:
            d9 = float('%.3f' % (np.abs(value - 0.25) / (0.3 - 0.2)))
            if d9 < 0.08:

                return False
            else:
                return True

        elif index == 10:
            d10 = float('%.3f' % (np.abs(value - 35.5) / (37.0 - 31.7)))
            if d10 < 0.04:
                return False
            else:
                return True

        elif index == 11:
            d11 = float('%.3f' % (np.abs(value - 1500) / (1650 - 1200)))
            if d11 < 0.02:
                return False
            else:
                return True

    # 根据当前代的情况计算下一代的区间和最大改变数量 test done
    def cal_next_range_maxNums(self, cur_paras, cur_gen_sol, or_sd, original_value, cur_range, cur_max_nums):
        # sd 升序orderby sd 小于 初始
        # print(len(cur_paras) + len(cur_gen_sol)+" do ****************************************")
        cur_gen = np.concatenate((np.array(cur_paras), np.array(cur_gen_sol)), axis=1)
        cur_list = cur_gen.tolist()
        cur_gen_sol.sort(reverse=False, key=lambda cur_gen_sol: cur_gen_sol[1])
        filter_sol = list(filter(lambda x: x[1] < or_sd, cur_gen_sol))
        # curr_paras 是上一轮的 50* 12 的二维数组 的data 需要分析所有para *50 的上下届
        # res_next_range 是一个 12*2 的二维数组
        # how to cal res_next_nums 对过滤后的数组各列取平均值 ，选取第i（3）个 为mid
        # res_next_nums = 12
        if len(filter_sol) == 0:
            res_next_nums = cur_max_nums
        else:
            mid = np.mean(filter_sol, 0)
            res_next_nums = int(round(mid[2], 0))
        #

        filter_cur = list(filter(lambda x: x[13] < or_sd, cur_list))

        res_next_range = cur_range
        if len(filter_cur) == 0:
            print("No found data : filter_cur < 2.2")
        else:
            for i in range(0, 12):
                positive_arr = []
                negative_arr = []
                for j in range(0, len(filter_cur)):
                    # 大于 初值的 平均值 做上届  小于初值的 平均值 做下届 不停的收敛更新
                    if filter_cur[j][i] > original_value[i]:
                        positive_arr.append(filter_cur[j][i])
                    if filter_cur[j][i] < original_value[i]:
                        negative_arr.append(filter_cur[j][i])
                pos_np = np.array(positive_arr)
                neg_np = np.array(negative_arr)
                if len(neg_np) != 0:
                    res_next_range[i][0] = round(np.mean(neg_np), 3)
                if len(pos_np) != 0:
                    res_next_range[i][1] = round(np.mean(pos_np), 3)
        return res_next_nums, res_next_range
        # test done

    def is_converge(self, all_comb):
        # all_comb 是一个n* (12+3)的二维数组

        # 归一化 sd 和 pc
        # all_comb = list(filter(lambda x: x[14] > 0, all_comb))
        np_al = np.array(all_comb)
        col_sd = np_al[:, 13]
        sd_max = np.max(col_sd)
        sd_min = np.min(col_sd)
        nums_max = 12
        nums_min = 1
        score = []
        for i in range(0, len(all_comb)):
            sum = []
            x = all_comb[i][12]
            #

            y = self.nor_single(all_comb[i][13], sd_min, sd_max)
            #
            z = self.nor_single(all_comb[i][14], nums_min, nums_max)
            sum.append(round((x + 6 * y + z) / 8, 4))
            score.append(sum)

        # 拼接 二维数组 n* (12+3)  n*(1)
        extra_list = np.concatenate((np.array(all_comb), np.array(score)), axis=1)
        # np.savetxt("extra" + str(len(all_comb)) + ".csv", extra_list, delimiter=",", fmt="%.4f")

        # 截取 最后 50*（12+3） 和 之前的 (n-50) *(12+3)
        cur_comb_np = extra_list[-self.pop_size:, :]

        pre_comb_np = extra_list[:(len(extra_list) - self.pop_size), :]

        # 升序 score 最小的排在前面

        cur_comb = cur_comb_np.tolist()
        # np.savetxt("./cur/cur" + str(len(all_comb)) + ".csv", cur_comb, delimiter=",", fmt="%.4f")
        pre_comb = pre_comb_np.tolist()
        print("cur_len:", len(cur_comb), " pre_len:", len(pre_comb), " all_len:", len(all_comb))

        cur_comb.sort(reverse=False, key=lambda cur_comb: cur_comb[15])
        #
        pre_comb.sort(reverse=False, key=lambda pre_comb: pre_comb[15])

        # 截取前50 个 最好的score
        best_comb = pre_comb[:self.pop_size]
        # np.savetxt("./best/best" + str(len(all_comb)) + ".csv", best_comb, delimiter=",", fmt="%.4f")

        # 降序 找到最差的 score
        best_comb.sort(reverse=True, key=lambda best_comb: best_comb[15])

        # 如果当前代中最好的 小于 之前所有解中最好的50中的最差的 就 还没收敛
        if cur_comb[0][15] < best_comb[0][15]:

            return False
        else:
            return True

    def cal_final_best(self, all_comb):
        # 归一化 sd 和 pc
        # all_comb = list(filter(lambda x: x[14] > 0, all_comb))
        np_al = np.array(all_comb)
        col_sd = np_al[:, 13]
        sd_max = np.max(col_sd)
        sd_min = np.min(col_sd)
        nums_max = 12
        nums_min = 1
        score = []
        for i in range(0, len(all_comb)):
            sum = []
            x = all_comb[i][12]
            #
            y = self.nor_single(all_comb[i][13], sd_min, sd_max)
            #

            z = self.nor_single(all_comb[i][14], nums_min, nums_max)
            sum.append(round((x + 6 * y + z) / 8, 4))
            score.append(sum)

        # 拼接 二维数组 n* (12+3)  n*(1)
        extra_list = np.concatenate((np.array(all_comb), np.array(score)), axis=1)
        all_merge = extra_list.tolist()
        all_merge.sort(reverse=False, key=lambda all_merge: all_merge[15])
        best_comb = all_merge[:self.pop_size]
        return best_comb

    # 归一化一个数组的值
    def nor(self, arr):
        np_arr = np.array(arr)
        max = np.max(np_arr)
        min = np.min(np_arr)
        nor_arr = (np_arr - min) / (max - min)
        return nor_arr.tolist()

    def nor_single(self, value, minimum, maximum):

        return (value - minimum) / (maximum - minimum)

    def search(self):
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
        # pickle_file  = open('nsga.pkl', 'rb')
        # al = pickle.load(pickle_file)
        # pickle_file.close()
        # algorithm = al
        # algorithm.getProblem().set_runner(self.runner)
        # eva = None
        # eva = algorithm.get_eva()
        # print("current eva:", eva)
        try:
            # print("abc")
            algorithm.run()
            # algorithm.rerun()
        except Exception as e:
            print("catch error:", e.args)
            save_file = open('nsga.pkl', 'wb')
            pickle.dump(algorithm, save_file)
            save_file.close()
            print("SAVE pkl")
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
        # algorithm = RandomSearch(
        #     problem=self.problem,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # )

        # algorithm.run()
        """
                   reload pickle
        """
        # #
        pickle_file = open('rs.pkl', 'rb')
        al = pickle.load(pickle_file)
        pickle_file.close()
        algorithm = al
        algorithm.getProblem().set_runner(self.runner)
        eva = algorithm.get_eva()
        print("current eva:", eva)
        eva = None
        try:
            # print("abc")
            # algorithm.run()
            algorithm.rerun()
        except Exception as e:
            print("catch error:", e.args)
            save_file = open('rs.pkl', 'wb')
            pickle.dump(algorithm, save_file)
            save_file.close()
            print("SAVE pkl")
            eva = algorithm.get_eva()
            print("current eva:", eva)
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
        # x0 = 5800
        # x1 = 0.15
        # x2 = 2.0
        # x3 = 0.35
        # x4 = 0.5
        # x5 = 10.0
        # x6 = 2404
        # x7 = 0.3
        # x8 = 3.5
        # x9 = 0.25
        # x10 = 35.5
        # x11 = 1500

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
        # physics = [5800.000, 0.150, 2.000, 0.350, 0.500, 10.000, 2404.00, 0.300, 3.500, 0.150, 35.500, 1500.000]
        physics = []
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
        if self.count == 36:
            print("time to end")
            self.runner = None
            raise Exception("over")
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
