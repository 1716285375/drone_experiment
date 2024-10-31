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


def vec3_magnitude(vec: Type[airsim.Vector3r]) -> float:
    return math.sqrt(vec.x_val ** 2 + vec.y_val ** 2 + vec.z_val ** 2)


# 打印log输出格式
def print_split():
    split = r""
    for i in range(100):
        split += r"#"
    print(split)


def print_output(variable):
    if type(variable) is not str:
        variable = str(variable)

    split = r""
    for i in range(50):
        split += r"-"
    split += 5 * r" "
    split += variable
    split += 5 * r" "
    for i in range(50):
        split += r"-"

    print(split)
