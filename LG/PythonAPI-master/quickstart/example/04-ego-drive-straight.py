#!/usr/bin/env python3
#
# Copyright (c) 2019-2021 LG Electronics, Inc.
#
# This software contains code licensed as described in LICENSE.
#

from environs import Env
import lgsvl
from prettytable import PrettyTable

print("Python API Quickstart #4: Ego vehicle driving straight")
env = Env()

sim = lgsvl.Simulator(env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host), env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
print("create sim")
print("sim cur",sim.current_scene)
print("lg:",lgsvl.wise.DefaultAssets.map_borregasave)
if sim.current_scene == lgsvl.wise.DefaultAssets.map_borregasave:
    print("reset")
    sim.reset()
else:
    print("load map")
    sim.load(scene="Bor")

spawns = sim.get_spawn()

state = lgsvl.AgentState()
state.transform = spawns[0]

forward = lgsvl.utils.transform_to_forward(spawns[0])

# Agents can be spawned with a velocity. Default is to spawn with 0 velocity
state.velocity = 20 * forward
# ego = sim.add_agent(env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.ego_lincoln2017mkz_apollo5), lgsvl.AgentType.EGO, state)
ego = sim.add_agent(name = "LinkAp5", agent_type = lgsvl.AgentType.EGO, state = state)
# The bounding box of an agent are 2 points (min and max) such that the box formed from those 2 points completely encases the agent
print("Vehicle bounding box =", ego.bounding_box)

print("Current time = ", sim.current_time)
print("Current frame = ", sim.current_frame)
table = PrettyTable()
s = ego.state
if s.angular_velocity is not None:
    table.add_column('参数', ['angular', 'speed', 'tip', 'mass', 'wheel_mass', 'wheel_radius', 'MaxMotorTorque',
                            'MaxBrakeTorque', 'MinRPM', 'MaxRPM', 'ShiftTime', 'TireDragCoeff', 'WheelDamping',
                            'TractionControlSlipLimit', 'MaxSteeringAngle', 'position'])
    table.add_column('值', [s.angular_velocity, s.velocity, s.tip, s.mass, s.wheel_mass, s.wheel_radius, s.MaxMotorTorque,
                           s.MaxBrakeTorque, s.MinRPM, s.MaxRPM, s.ShiftTime, s.TireDragCoeff, s.WheelDamping,
                           s.TractionControlSlipLimit, s.MaxSteeringAngle, s.position])
    print(table)
    print("-----------------------------------------------")
    print("angular:",s.angular_velocity )
    print("speed:",s.velocity)
    print("mass:",s.mass)
    print("tip:",s.tip)
    print("wheel", s.wheel_mass)
    print("wheel_radius", s.wheel_radius)
    print("MaxMotorTorque",s.MaxMotorTorque )
    print("MaxBrakeTorque", s.MaxBrakeTorque)
    print("MinRPM", s.MinRPM)
    print("MaxRPM", s.MaxRPM)
    print("ShiftTime", s.ShiftTime)
    print("TireDragCoeff", s.TireDragCoeff)
    print("WheelDamping", s.WheelDamping)
    print("TractionControlSlipLimit", s.TractionControlSlipLimit)
    print("MaxSteeringAngle", s.MaxSteeringAngle)
    print("distance", s.transform.position)
    print("-----------------------------------------------")

input("Press Enter to drive forward for 2 seconds")

# The simulator can be run for a set amount of time. time_limit is optional and if omitted or set to 0, then the simulator will run indefinitely
sim.run(time_limit=2.0)
#distance Vector(154.719970703125, -4.46302461624146, -63.7233467102051)

s = ego.state
s.mass = 2500
s.tip = 80
s.wheel = 30
s.wheel_radius = 0.35

# s.TireDragCoeff = 4
# s.WheelDamping = 160
ego.state = s
print("distance", s.transform.position)
print("Current time = ", sim.current_time)
print("Current frame = ", sim.current_frame)


input("Press Enter to continue driving for 2 seconds")

sim.run(time_limit=10.0)


s1 = ego.state
if s1.angular_velocity is not None:
    print("-----------------------------------------------")
    print("speed2:",s1.velocity)
    print("mass2:", s1.mass)
    print("tip2:", s1.tip)
    print("wheel2", s1.wheel_mass)
    print("wheel_radius2", s1.wheel_radius)
    print("MaxMotorTorque", s1.MaxMotorTorque)
    print("MaxBrakeTorque", s1.MaxBrakeTorque)
    print("MinRPM", s1.MinRPM)
    print("MaxRPM", s1.MaxRPM)
    print("ShiftTime", s1.ShiftTime)
    print("TireDragCoeff", s1.TireDragCoeff)
    print("WheelDamping", s1.WheelDamping)
    print("TractionControlSlipLimit", s1.TractionControlSlipLimit)
    print("MaxSteeringAngle", s1.MaxSteeringAngle)
    print("distance", s1.transform.position)
    print("-----------------------------------------------")

print("Current time = ", sim.current_time)
print("Current frame = ", sim.current_frame)
