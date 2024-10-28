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


air_data = {}
client = airsim.MultirotorClient()
client.confirmConnection()
for i, (key, value) in enumerate(drone_cfg.items()):
    for epoch in range(10):
        client.enableApiControl(True, vehicle_name=key)
        client.armDisarm(True, vehicle_name=key)
        client.takeoffAsync(vehicle_name=key)
        client.moveToZAsync(-3, 1, vehicle_name=key).join()
        # client.simFlushPersistentMarkers()
        data = []
        for j in range(10):
            vx = random.uniform(-1, 1)
            vy = random.uniform(-1, 1)
            vz = random.uniform(-1, 1)

            state = client.simGetGroundTruthKinematics(vehicle_name=key)
            pos = [state.position.x_val, state.position.y_val, state.position.z_val]

            data.append(pos)

            client.moveByVelocityAsync(vx, vy, vz, 1).join()

            if client.getMultirotorState().collision.has_collided:
                break
        client.reset()

        air_data[epoch] = data
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    file_name = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
    with open(f"./data/positions/{file_name}.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(air_data))

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

