import airsim
import json
import random
import time
import numpy as np
from common.airsim_utils import *

cfg = json.load(open(r"C:\Users\jie\Documents\AirSim\settings.json", encoding='utf-8'))
print(cfg)
client = airsim.MultirotorClient()
client.confirmConnection()


drones = cfg["Vehicles"]
print(drones)
keys = [key for key in drones.keys()]
client.reset()
client.enableApiControl(True, vehicle_name=keys[0])
client.armDisarm(True, vehicle_name=keys[0])

client.takeoffAsync(vehicle_name=keys[0]).join()

# client.moveToZAsync(-20, 10, vehicle_name=keys[0]).join()
client.moveToZAsync(-20, 10, vehicle_name=keys[0]).join()

data = {}
for i in range(20):

    state = client.getMultirotorState().kinematics_estimated.orientation
    print(f"第 {i}: 轮 ", state)

    angle = airsim.to_eularian_angles(state)
    print(angle)
    yaw = angle[2]
    print("弧度: ", yaw)
    yaw_angle = yaw * (180 / np.pi)
    print("角度: ", yaw_angle)
    value = random.randint(-20, 20)
    print("偏移值:", value)

    client.rotateToYawAsync(yaw_angle + value, timeout_sec=2, vehicle_name=keys[0]).join()
    time.sleep(2)
    print("偏移: ", yaw_angle + value)
    client.moveByVelocityZAsync(10, 0, -20, 2).join()


mode = airsim.YawMode(
    is_rate=False,
    yaw_or_rate=1.57
)
# x = 0
# y = 0
# z = -20
# yaw = [36, 72, 108, 144, 180, 216, 252, 288, 324, 360]
# for i in range(10):
#     x += 1
#     y += 1
#     state = client.getMultirotorState(vehicle_name=keys[0]).kinematics_estimated.orientation
#     angle = airsim.to_eularian_angles(state)
#     # yaw = angle[0]
#     print_split()
#     client.rotateToYawAsync(yaw[i], vehicle_name=keys[0]).join()
#     # client.moveByVelocityZAsync(2, 2, -20, 1, vehicle_name=keys[0]).join()
#     client.moveToPositionAsync(x, y, z, 1, vehicle_name=keys[0]).join()
