import airsim
import typing
import math
from typing import Type


def distance(vec1: Type[airsim.Vector3r], vec2: Type[airsim.Vector3r]) -> float:
    dis = math.sqrt((math.pow(vec1.x_val - vec2.x_val, 2) +
                     math.pow(vec1.y_val - vec2.y_val, 2) +
                     math.pow(vec1.z_val - vec2.z_val, 2)))

    return dis


def vec3_sub(vec1: Type[airsim.Vector3r], vec2: Type[airsim.Vector3r]) -> airsim.Vector3r:
    return airsim.Vector3r(vec1.x_val - vec2.x_val, vec1.y_val - vec2.y_val, vec1.z_val - vec2.z_val)


def print_split():
    split = r""
    for i in range(100):
        split += r"#"
    print(split)
