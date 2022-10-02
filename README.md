# 基于手部骨架姿态识别的增强现实交互系统

## 介绍

第十四届大学生计算机设计大赛参赛作品。

4.16 初赛提交。

5.22 省赛答辩。

5.23 省赛二等奖。

5.30 国赛提交。

8.15 国赛答辩。

8.17 国赛二等奖。


## 使用说明

1. handpose_setter.py：可以设置自定义的手势类型。
2. inference.py：运行项目主体推理。
3. 模型权重与素材文件: [3fui](https://pan.baidu.com/s/1hJrEA3KU9_VKYF6O_eFB4Q)
4. 展示视频: https://www.bilibili.com/video/BV1H64y1y7X1/

## 思路概述

1. 轻量级的 YOLOv5 手部目标检测
2. 简单 IOU 手部目标追踪
3. ResNet50 手部关键点识别
4. 关键点的二维向量夹角定义手势语义
5. 时序逻辑上的运动轨迹分析
6. 手势语义与运动轨迹结合实现交互

## 参考

1.  YOLOv5: https://github.com/ultralytics/yolov5
2.  关键点检测: https://codechina.csdn.net/EricLee/handpose_x
