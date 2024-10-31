import airsim
import json

airsim_cfg = json.load(open(r"C:\Users\jie\Documents\AirSim\settings.json", encoding='utf-8'))
drone_name = {}
drone_id2key = {}
for i, (key, value) in enumerate(airsim_cfg["Vehicles"].items()):
    drone_name[key] = value
    drone_id2key[i] = key

print(drone_name)
print(drone_id2key)

leader = airsim.MultirotorClient()
leader.confirmConnection()

follower = airsim.MultirotorClient()
follower.confirmConnection()

leader.reset()
follower.reset()

leader.enableApiControl(True, vehicle_name=drone_id2key[0])
leader.armDisarm(True, vehicle_name=drone_id2key[0])

follower.enableApiControl(True, vehicle_name=drone_id2key[1])
follower.armDisarm(True, vehicle_name=drone_id2key[1])

takeoff_height = -20

leader.moveToZAsync(takeoff_height, 5, vehicle_name=drone_id2key[0])  # 移动到指定的位置：（x, y, z），速度为2m/s
follower.moveToZAsync(takeoff_height, 5, vehicle_name=drone_id2key[1]).join()


leader.reset()
follower.reset()


leader.enableApiControl(True, vehicle_name=drone_id2key[0])
leader.armDisarm(True, vehicle_name=drone_id2key[0])

follower.enableApiControl(True, vehicle_name=drone_id2key[1])
follower.armDisarm(True, vehicle_name=drone_id2key[1])

takeoff_height = -20

leader.moveToZAsync(takeoff_height, 5, vehicle_name=drone_id2key[0])  # 移动到指定的位置：（x, y, z），速度为2m/s
follower.moveToZAsync(takeoff_height, 5, vehicle_name=drone_id2key[1]).join()

leader.reset()
follower.reset()