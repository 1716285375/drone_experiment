import json
import numpy as np
import math
import time
import random
import airsim
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


client = airsim.MultirotorClient()
client.confirmConnection()
for i, (key, value) in enumerate(drone_cfg.items()):
    for epoch in range(10):
        client.enableApiControl(True, vehicle_name=key)
        client.armDisarm(True, vehicle_name=key)
        client.takeoffAsync(vehicle_name=key)
        client.moveToZAsync(-20, 5, vehicle_name=key).join()
        data = []
        # 航向角偏移， 绕Z轴旋转
        # yaw = 0
        # for j in range(10):
        #     yaw += 18
        #     print(yaw)
        #     client.rotateToYawAsync(yaw, vehicle_name=key).join()   # 顺时针转

        # 按照速率偏移
        # v = 18
        # for j in range(10):
        #     client.rotateByYawRateAsync(v, 1, vehicle_name=key).join()
        roll = 0
        z = -20
        for j in range(4):
            roll += 45
            z -= 3
            client.moveByRollPitchYawZAsync(roll, 0, 0, z, 1, vehicle_name=key).join()
        client.reset()
    # for i, (key, value) in enumerate(drone_cfg.items()):
    #     # 悬停
    #     client.hoverAsync().join()
    #
    #     # 着陆
    #     client.landAsync(vehicle_name=key).join()
    #
    #     # lock
    #     client.armDisarm(False, vehicle_name=key)
    #     # release control
    #     client.enableApiControl(False, vehicle_name=key)



# rotateToYawAsync