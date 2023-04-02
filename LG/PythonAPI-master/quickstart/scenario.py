# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 15:23
# @Author  : panqi
# @File    : scenario.py
import json
import math
import os
import socket

import lgsvl
from environs import Env
from prettytable import PrettyTable
from lgsvl.geometry import Vector, Transform
import time
from func_timeout import func_set_timeout, FunctionTimedOut


def main(phy, env, sim, ss):
    print("into load")
    if sim.current_scene == lgsvl.wise.DefaultAssets.map_borregasave:
        print("load over")
        sim.reset()
        pass
    else:
        sim.load(lgsvl.wise.DefaultAssets.map_borregasave)
    spawns = sim.get_spawn()


    # npc stop 3.25--------------------------------------
    print("load npc")
    state_npc = lgsvl.AgentState()
    state_npc.transform = spawns[0]
    state_npc.transform.position = Vector(165.646987915039, -4.49, -63)
    npc = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state_npc)
    state_npc.transform.position = Vector(165.646987915039, -4.49, -60)
    npc_1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state_npc)

    # ------------------------------------------------------
    print("load agent")
    state = lgsvl.AgentState()
    state.transform = sim.map_point_on_lane(Vector(100.646987915039, -4.49, -45))
    ego = sim.add_agent(lgsvl.wise.DefaultAssets.ego_lincoln2017mkz_apollo5, lgsvl.AgentType.EGO, state)

    ego.connect_bridge(
        env.str("LGSVL__AUTOPILOT_0_HOST", lgsvl.wise.SimulatorSettings.bridge_host),
        env.int("LGSVL__AUTOPILOT_0_PORT", lgsvl.wise.SimulatorSettings.bridge_port)
    )
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

    # Dreamview setup
    dv = lgsvl.dreamview.Connection(sim, ego, env.str("LGSVL__AUTOPILOT_0_HOST", "192.168.50.51"))
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
    destination = spawns[0].destinations[0]
    # print("des:",destination)
    print("ready to setup apollo")
    try:
        dv.setup_apollo(destination.position.x, destination.position.z, modules)
    except Exception as e:
        while not ego.bridge_connected:
            time.sleep(1)
        try:
            dv.reconnect()
            dv.setup_apollo(destination.position.x, destination.position.z, modules)
        except Exception as e:
            raise Exception("dv time out,not bridge")
        raise Exception("dv time out")


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

    e = ego.state
    table_3 = PrettyTable()
    if e is not None:
        table_3.add_column('参数', ['angular', 'speed', 'tip', 'mass', 'wheel_mass', 'wheel_radius', 'MaxMotorTorque',
                                  'MaxBrakeTorque', 'MinRPM', 'MaxRPM', 'ShiftTime', 'TireDragCoeff', 'WheelDamping',
                                  'TractionControlSlipLimit', 'MaxSteeringAngle', 'position'])
        table_3.add_column('值',
                           [e.angular_velocity, e.velocity, e.tip, e.mass, e.wheel_mass, e.wheel_radius,
                            e.MaxMotorTorque,
                            e.MaxBrakeTorque, e.MinRPM, e.MaxRPM, e.ShiftTime, e.TireDragCoeff, e.WheelDamping,
                            e.TractionControlSlipLimit, e.MaxSteeringAngle, e.position])
        print(table_3)
    #     del table_3

    print("ready to run!")
    speed = []
    infos = []
    dis = []
    time_slip = 32
    car_length = 4.7
    try:

        ss.send(json.dumps(['start']).encode('utf-8'))
        sim.run(time_limit=16)

        # ---------------------------------------------------

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
        obstacle_arr = received_data['obstacle_arr']
        traffic_light = received_data['traffic_light']

        ACC = []
        ACC2 = []
        temp = 0
        temp_acc = 0
        for i in range(len(state_arr)):
            if state_arr[i]['speed'] != 0.0 and state_arr[i]['brake'] > 0.0:
                ACC.append(state_arr[i]['acceleration'])
                if i != 0:
                    temp = state_arr[i]['speed'] - state_arr[i - 1]['speed']
                    temp_acc = float(temp) / 0.5
                    if temp_acc < 0:
                        ACC2.append(temp_acc)
        average_acc = (sum(ACC2) / len(ACC2))

        npc_pos_y = 4141411.35545349
        num_ttc = 0
        TIT = 0
        for i in range(len(pose_arr)):
            velocity = pose_arr[i]['vx'] ** 2
            velocity += pose_arr[i]['vy'] ** 2
            velocity += pose_arr[i]['vz'] ** 2
            v_finally = math.sqrt(velocity)
            rela_loc = pose_arr[i]['py'] - npc_pos_y
            ttc = -1
            if v_finally != 0:
                ttc = (rela_loc -car_length) / v_finally
                if ttc < 3:
                    num_ttc += 1
                    temp = 3 - ttc
                    TIT += (temp * 0.2)

                TET = float('%.3f' % (num_ttc * 0.2))

        # ss.close()



    except FunctionTimedOut:
        raise Exception("sim time out")

    distance = npc.state.transform.position.x - ego.state.transform.position.x

    return distance,TET,TIT,average_acc
