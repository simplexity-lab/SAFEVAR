from testing_LG import LG
import test as lg
import lgsvl
from prettytable import PrettyTable
from environs import Env

if __name__ == '__main__':
    lg = LG()
    for i in range(2):
        lg.create_ego_and_connect(1,2)
        # lg.get_physics()
        phy = [2500, 30, 0.32, 8299, 800, 3000, 450, 39.4, 4, 1, 0.4, 0.8]
        lg.set_npc_follow()
        lg.set_physics(phy)
        result = lg.run()
        lg.destory()
    # env = Env()
    # sim = lgsvl.Simulator(env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host),
    #                       env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
    # for i in range(5):
    #     lg.main(sim)
