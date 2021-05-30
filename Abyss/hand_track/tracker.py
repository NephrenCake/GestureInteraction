# -- coding: utf-8 --
import _thread
import json

import cv2
import numpy as np

from Abyss.hand_track.angle_util import pose_to_angles, piano_judge
from Abyss.hand_track.draw_util import draw
from Abyss.interaction.audio_thread import play_piano


class Tracker:
    """
    检测跟踪模块。
    对深度学习模型的结果进行初步处理。
    1. 增强鲁棒性
    2. 识别手势
    3. IOU跟踪
    """

    def __init__(self, pose_cfg, pose_thres=0.2, no_data_limit=5):
        # 设置
        with open(pose_cfg, 'r') as f:
            self.pose_list = json.load(f)  # 手势规则 list[dict{name:int,angle:list[float*5]}*n个手势]
        self.pose_thres = pose_thres
        self.no_data_limit = no_data_limit  # 容忍检测丢失的帧数

        # 缓存
        self.last_box = np.array([[0, 0, 0, 0],
                                  [0, 0, 0, 0]])  # 上一帧的box位置
        self.no_data_now = 0  # 当前检测丢失累计
        self.active_click = np.array([[0, 0, 0],
                                      [0, 0, 0]])  # 点击响应位置+静态手势
        self.plot_cache = [[None, None],
                           [None, None]]

        # 多线程
        self.piano_data = []
        _thread.start_new_thread(play_piano, (self.piano_data,))
        self.circle_list = []

    def update(self, det, key_point_list):
        # 返回格式
        for n, (*xyxy, conf, cls) in enumerate(det):  # 对可能有两只手进行遍历，六个参数
            x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            iou: np.ndarray = self.__compute_iou(x1, y1, x2, y2)  # 获得一个二维数组，分别对应id0/1的手势的iou
            track_id, iou_val = iou.argmax(), iou.max()  # 获得iou最大对应的手 初步追踪
            pose: int = self.__compute_pose(key_point_list[n])

            # piano
            piano = False
            if piano:
                self.circle_list = piano_judge(key_point_list[n], self.piano_data)

            # 更新内部追踪
            if iou_val == 0:  # 当前手对于之前的两只手都匹配不上
                if self.last_box[track_id].max() != 0:  # 当前追踪的一只手有记录
                    if self.last_box[1 - track_id].max() != 0:  # 记录的另一只手也有记录
                        self.update_nodata([0, 1])  # ①移动过快：全部重置
                        return
                    else:  # 记录的另一只手没有记录
                        track_id = 1 - track_id  # ②画面中加入新手，修正id

            self.no_data_now = 0  # 重置检测丢失计数

            # ③成功追踪到正在移动的手
            self.last_box[track_id] = np.array([x1, y1, x2, y2])
            self.active_click[track_id][0] = key_point_list[n][8][0]
            self.active_click[track_id][1] = key_point_list[n][8][1]
            self.active_click[track_id][2] = pose

            self.plot_cache[track_id][0] = np.array([x1, y1, x2, y2, conf, track_id, iou_val, pose], dtype=np.float32)
            self.plot_cache[track_id][1] = key_point_list[n]

            if len(det) == 1:
                self.update_nodata(1 - track_id, now=True)

    def plot(self, im0):
        for track_id in range(0, 2):
            if self.plot_cache[track_id][0] is not None:
                draw(im0, self.plot_cache[track_id][0], self.plot_cache[track_id][1])
        for i, point in enumerate(self.circle_list):
            x = int(point[0])
            y = int(point[1])
            c = int(255 * (i + 1) / 6)
            cv2.circle(im0, (x, y), 25, (c, 255 - c, abs(122 - c)), 5)

    def get_order(self):
        """
        获得响应位置及手势
        """
        return self.active_click

    def update_nodata(self, idx, now=False):
        """
        清空记录数据
        """
        if now or self.no_data_now == self.no_data_limit:
            self.last_box[idx] = np.array([0, 0, 0, 0])
            self.active_click[idx] = np.array([0, 0, 0])
            if idx == 1 or idx == 0:
                self.plot_cache[idx][0] = None
                self.plot_cache[idx][1] = None
            else:
                self.plot_cache = [[None, None],
                                   [None, None]]
            self.no_data_now = 0
        else:
            self.no_data_now += 1

    def __compute_iou(self, x1, y1, x2, y2):
        """
        计算当前预测框，与记录的两个预测框的IOU值
        """
        box1 = np.array([x1, y1, x2, y2])
        iou_list = []
        for box2 in self.last_box:
            h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
            w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
            area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
            area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
            inter = w * h
            union = area_box1 + area_box2 - inter
            iou = inter / union
            iou_list.append(iou)

        iou_list = np.array(iou_list)
        return iou_list

    # 识别不出手势就是0
    def __compute_pose(self, key_point):
        """
        读取设置文件，匹配手势
        """
        angles = pose_to_angles(key_point)  # [    0.99953    -0.91983    -0.95382    -0.98989    -0.99999]
        for pose in self.pose_list:
            max = (np.array(pose['angle']) + self.pose_thres >= angles).sum()
            min = (np.array(pose['angle']) - self.pose_thres <= angles).sum()
            if max == min == 6:
                return int(pose['name'])
        return 0


if __name__ == '__main__':
    t = Tracker()
    # t.last_time[0] = 90
    # t.last_time[1] = 90
    # t.get_order()
    # t.update_nodata([0, 1])
    # t.__compute_iou(1, 1, 5, 5)
    # t.last_box += 1
    # print(t.last_box)
    # print(t.last_time[0])
    # print(n)
    # print(type(n))
