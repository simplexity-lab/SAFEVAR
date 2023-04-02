

from testing_LG import LG


if __name__ == '__main__':
    lg = LG()
    for i in range(5):
        lg.create_ego_and_connect()
        # lg.get_physics()
        lg.set_npc_follow()
        phy = [2500, 30, 0.32, 8299, 800, 3000, 450, 39.4, 4, 1, 0.4, 0.8]
        lg.set_physics(phy)
        result = lg.run()
        lg.destory()