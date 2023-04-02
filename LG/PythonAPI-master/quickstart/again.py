# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 11:25
# @Author  : Chengjie
# @File    : again.py
import json
import math
import os
import lgsvl
from environs import Env
from prettytable import PrettyTable
from lgsvl.geometry import Vector, Transform
import time
from func_timeout import func_set_timeout, FunctionTimedOut


def main(phy, env, sim, ego, npc, ss,step):
    if sim.current_scene == lgsvl.wise.DefaultAssets.map_borregasave:
        # sim.reset()
        pass
    else:
        sim.load(lgsvl.wise.DefaultAssets.map_borregasave)
    spawns = sim.get_spawn()

    # bridge connection
    print("Waiting for connection...")
    timeout = 60
    elapsed = 0
    while not ego.bridge_connected:
        time.sleep(1)
        # elapsed = elapsed + 1
        # if elapsed >= timeout:
        #     raise Exception("bridge unconnected")
    print("Bridge connected:", ego.bridge_connected)

    x = ego.state
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
    ego.state = x


    col_flag = False
    col_ve = 0
    def on_collision(agent1, agent2, contact):
        # name1 = vehicles[agent1]
        nonlocal col_flag
        col_flag = True
        nonlocal col_ve
        collsion_velocity = agent1.state.velocity
        col_ve = collsion_velocity
        f = open("collision.txt", 'a')
        f.write(str(step)+":"+str(phy))
        f.write(" "+str(collsion_velocity))
        f.write("\n")
        f.close()
    ego.on_collision(on_collision)
    print("ready to run!")
    try:
        ss.send(json.dumps(['start']).encode('utf-8'))
        sim.run(time_limit=16.0)
        # --handle data ----------------------------------------
        ss.send(json.dumps(['stop']).encode('utf-8'))
        cmd_res_size = ss.recv(1024)
        length = int(cmd_res_size.decode())
        ss.send(json.dumps(['confirmed']).encode('utf-8'))
        received_size = 0
        received_data = b''
        while received_size < length:
            cmd_res = ss.recv(1024)
            received_size += len(cmd_res)
            received_data += cmd_res
        received_data = json.loads(received_data.decode('utf-8'))
        state_arr = received_data['state_arr']
        pose_arr = received_data['pose_arr']
        # -------------------------------------------------------------
        ACC2 = []
        temp = 0
        temp_acc = 0
        for i in range(len(state_arr)):
            if state_arr[i]['speed'] != 0.0 and state_arr[i]['brake'] > 0.0:
                if i != 0:
                    temp = state_arr[i]['speed'] - state_arr[i - 1]['speed']
                    temp_acc = float(temp) / 0.5
                    if temp_acc < 0:
                        ACC2.append(temp_acc)
        average_acc = (sum(ACC2) / len(ACC2))

        # ------------------------------------------------------------
        # TTC
        npc_pos_y = 4141411.35545349
        TET = 0
        TIT = 0
        num_ttc = 0
        for i in range(len(pose_arr)):
            velocity = pose_arr[i]['vx'] ** 2
            velocity += pose_arr[i]['vy'] ** 2
            velocity += pose_arr[i]['vz'] ** 2
            v_finally = math.sqrt(velocity)
            rela_loc = pose_arr[i]['py'] - npc_pos_y
            ttc = -1
            if v_finally != 0:
                ttc = (rela_loc - 4.7) / v_finally
                if ttc < 3:
                    num_ttc += 1
                    temp = 3 - ttc
                    TIT += (temp * 0.2)

        TET = num_ttc * 0.2
        # ----------------------------------------------------------


    except FunctionTimedOut:
        raise Exception("sim time out")
    if col_flag:
        ve = col_ve.x **2
        ve += col_ve.y **2
        ve += col_ve.z **2
        distance = -math.sqrt(ve)
        f = open("collision.txt", 'a')
        dis = npc.state.transform.position.x - ego.state.transform.position.x
        f.write(str(step)+":"+str(dis))
        f.write("\n")
        f.close()
    else:
        distance = npc.state.transform.position.x - ego.state.transform.position.x

    # sim.reset()

    return distance, float('%.3f' % TET), float('%.3f' % TIT), float('%.3f' % average_acc)
