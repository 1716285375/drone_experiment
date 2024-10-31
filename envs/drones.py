import airsim


class Drone:
    def __init__(self, ip="127.0.0.1", counts=2):
        self.ip = ip
        self.counts = counts
        self.clients = []
        self.load_airsim()

    def load_airsim(self):
        for i in range(self.counts):
            client = airsim.MultirotorClient()
            client.confirmConnection()
            self.clients.append(client)


# if __name__ == '__main__':
#
#     # to test if can fly with multiple airsim client
#     drone = Drone(ip="127.0.0.1", counts=2)
#     drone.clients[0].enableApiControl(True, vehicle_name="UAV1")
#     drone.clients[1].enableApiControl(True, vehicle_name="UAV2")
#
#     drone.clients[0].armDisarm(True)
#     drone.clients[1].armDisarm(True)
#
#     drone.clients[0].moveToZAsync(-20, 5, vehicle_name="UAV1")
#     drone.clients[1].moveToZAsync(-20, 5, vehicle_name="UAV2").join()
#
#     drone.clients[0].reset()
#     drone.clients[1].reset()
