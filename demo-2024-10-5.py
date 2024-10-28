import airsim
import json
import numpy as np
import math
import time
import random
from matplotlib import pyplot as plt

setting_json = r"C:\Users\jie\Documents\AirSim\settings.json"
with open(setting_json, 'r', encoding='utf-8') as f:
    setting_dict = json.load(f)


# print(setting_dict)

drone_cfg = setting_dict['Vehicles']
# print(drone_cfg)
# print(len(drone_cfg))
del drone_cfg["UAV2"]
del drone_cfg["UAV3"]
del drone_cfg["UAV4"]
del drone_cfg["UAV5"]

if drone_cfg:
    vehicle_type = 'VehicleType'
    x = 'X'
    y = 'Y'
    z = 'Z'
    yaw = 'Yaw'
    pitch = 'Pitch'
    roll = 'Roll'

    drones = {}
    for key, value in drone_cfg.items():
        # print(key, value)
        drones[key] = value
    # print(drones)
    for key in drones:
        # print(key)
        # print(drones[key][x], drones[key][y], drones[key][z])
        pass
else:
    print("doesn't exist the drone!")

DEBUG = True
if DEBUG:
    client = airsim.MultirotorClient()  # connect the airsim
    client.confirmConnection()

    print("---------------------SPLIT LINE--------------------------")
    origin_pos = np.zeros((len(drone_cfg), 3))
    # print(origin_pos)

    # 启动无人机
    for i, (key, value) in enumerate(drone_cfg.items()):
        # print(i, key, value)
        name = key
        client.enableApiControl(True, vehicle_name=key)
        client.armDisarm(True, vehicle_name=key)

        state = client.getMultirotorState(vehicle_name=key)
        print(state)

        pos_s = client.getMultirotorState(vehicle_name=key).kinematics_estimated.position
        print(pos_s)

        if i != len(drone_cfg) - 1:
            client.takeoffAsync(vehicle_name=key)
        else:
            client.takeoffAsync(vehicle_name=key).join()
        pos_e = client.getMultirotorState(vehicle_name=key).kinematics_estimated.position
        print(pos_e)

    # 控制无人机飞到同一高度
    for i, (key, value) in enumerate(drone_cfg.items()):
        name = key
        if i != len(drone_cfg) - 1:
            client.moveToZAsync(-3, 1, vehicle_name=key)
        else:
            client.moveToZAsync(-3, 1, vehicle_name=key).join()

    #
    motion2idx = {
        0: "Linear",
        1: "Left",
        2: "Right",
        3: "Up",
        4: "Down",
        5: "Acc",
        6: "Dec",
        7: "Acc-Dec"
    }
    motion_mode = {
        'Linear': [1, 2, 0],
        'Left': [-math.pi / 2, 0],
        'Right': [0, math.pi / 2],
        'Up': [0, math.pi / 2],
        'Down': [-math.pi / 2, 0],
        'side-left': [-math.pi / 2, 0],
        'side-right': [0, math.pi / 2],
        'Acc': [1, 2, 0],
        'Dec': [1, 2, 0],
        'Acc-Dec': [1, 2, 0]
    }
    for i, (key, value) in enumerate(drone_cfg.items()):
        # client.moveByRollPitchYawrateZAsync(math.pi / 6, 0, 0, -15, 2, vehicle_name=key).join()
        client.moveByRollPitchYawrateZAsync(-math.pi / 6, 0, 0, -10, 2, vehicle_name=key).join()
        time.sleep(5)
        client.reset()
