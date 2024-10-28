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

    for i, (key, value) in enumerate(drone_cfg.items()):

        state = client.simGetGroundTruthKinematics(vehicle_name=key)
        pos_reserve = np.array([[state.position.x_val], [state.position.y_val], [state.position.z_val]])
        for j in range(2000):

            state = client.simGetGroundTruthKinematics(vehicle_name=key)
            pos = np.array([[state.position.x_val], [state.position.y_val], [state.position.z_val]])

            client.moveByVelocityAsync(0, 0, -0.2, 1).join()

            point_reserve = [airsim.Vector3r(pos_reserve[0, 0], pos_reserve[1, 0], pos_reserve[2, 0])]
            point = [airsim.Vector3r(pos[0, 0], pos[1, 0], pos[2, 0])]
            # point_end = [airsim.Vector3r(pos[])]
            print("point_reserve:", point_reserve)
            print("point:", point)
            print("-------------------------------------------------------")
            client.simPlotLineList(point_reserve + point, color_rgba=(0.0, 1.0, 0.0, 1.0), is_persistent=True)
            # client.simFlushPersistentMarkers()

    for i, (key, value) in enumerate(drone_cfg.items()):
        # 悬停
        client.hoverAsync().join()

        # 着陆
        client.landAsync(vehicle_name=key).join()

        state = client.getMultirotorState(vehicle_name=key)
        print("---------collision info-----------")
        print(state.collision)
        print("---------collision info-----------")

        # lock
        client.armDisarm(False, vehicle_name=key)
        # release control
        client.enableApiControl(False, vehicle_name=key)
        # time.sleep(1)
        # client.reset()
