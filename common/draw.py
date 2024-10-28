from matplotlib import pyplot as plt
import numpy as np
import os
import json
import random


def random_hex_color():
    def r():
        return random.randint(0, 255)

    return '#%02X%02X%02X' % (r(), r(), r())


root = r"../data/positions"
with open(os.path.join(root, "2024-10-07_19-20-40.json"), "r", encoding='utf-8') as f:
    data = json.load(f)

# print(data)

# print(len(data['0']))
x = {}
y = {}
z = {}
for key, value in data.items():
    # print(key, value)
    data_x = []
    data_y = []
    data_z = []
    for v in value:
        data_x.append(v[0])
        data_y.append(v[1])
        data_z.append(v[2])
    x[key] = data_x
    y[key] = data_y
    z[key] = data_z

colors = {}
for i in range(len(x)):
    colors[str(i)] = random_hex_color()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for key in x.keys():
    ax.plot(x[key], y[key], z[key], color=colors[key])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# ax.set(xticklabels=[],
#        yticklabels=[],
#        zticklabels=[])

plt.show()

