# from testing_LG import LG
# from conbim_LGSVL import LG
# import testingApollo as lg
import json
import socket

import lgsvl
import scenario as lg
import again as new_lg
from prettytable import PrettyTable
from environs import Env

if __name__ == '__main__':

    # lg = LG()
    # for i in range(1):
    #     lg.create_ego_and_connect(1,2)
    #     # lg.get_physics()
    #     # phy = [2500, 30, 0.32, 8299, 800, 3000, 450, 39.4, 4, 1, 0.4, 0.8]
    #     # lg.set_npc_follow()
    #     # lg.set_physics(phy)
    #     lg.set_Pedestrian()
    #     result = lg.run()
    #     lg.destory()

    env = Env()
    sim = lgsvl.Simulator(env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host),
                          env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
    HOST = '192.168.50.51'
    PORT = 6007
    ADDR = (HOST, PORT)
    ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ss.connect(ADDR)

    phy = [2120.0, 30.0, 0.35, 8299.0, 800.0, 3000.0, 450.0, 39.4 ,4.0, 1.0, 0.4, 0.8]

    for i in range(1):
        r, tet, tit, aveDece = lg.main(phy, env, sim, ss)
        # result, TET, TIT, average_acc = lg.main(phy, env, sim, ego, npc, ss)
        print("distance:", r)
