# -- coding: utf-8 --
import numpy as np


def vectors_to_angle(p1, p2, p3, p4):
    """
    p1-p2,p3-p4两向量夹角
    返回的是cos a [0,1]
    """
    x1 = p1[0] - p2[0]
    y1 = p1[1] - p2[1]
    x2 = p3[0] - p4[0]
    y2 = p3[1] - p4[1]
    a = x1 * x2 + y1 * y2
    b = ((x1 ** 2 + y1 ** 2) ** 0.5) * ((x2 ** 2 + y2 ** 2) ** 0.5)
    angle = a / b
    return angle


def pose_to_angles(key_point):
    """

    :param key_point: 21个关键点坐标
    :type key_point:
    :return: 6个cos值
    :rtype:
    """
    a1 = vectors_to_angle(key_point[0], key_point[1], key_point[3], key_point[4])
    a2 = vectors_to_angle(key_point[0], key_point[5], key_point[6], key_point[8])
    a3 = vectors_to_angle(key_point[0], key_point[9], key_point[10], key_point[12])
    a4 = vectors_to_angle(key_point[0], key_point[13], key_point[14], key_point[16])
    a5 = vectors_to_angle(key_point[0], key_point[17], key_point[18], key_point[20])
    a6 = vectors_to_angle(key_point[0], key_point[4], key_point[0], key_point[8])  # 大拇指与食指
    return np.array([a1, a2, a3, a4, a5, a6], dtype=np.float32)


def get_dis(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def piano_judge(key_point, piano_data):
    m = 676
    circle = []
    if get_dis(key_point[7], key_point[8]) < m:
        piano_data.append("1")
        circle.append(key_point[8])
    elif get_dis(key_point[11], key_point[12]) < m:
        piano_data.append("2")
        circle.append(key_point[12])
    elif get_dis(key_point[15], key_point[16]) < m:
        piano_data.append("3")
        circle.append(key_point[16])
    elif get_dis(key_point[19], key_point[20]) < m:
        piano_data.append("4")
        circle.append(key_point[20])

    return circle
