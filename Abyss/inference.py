# -- coding: utf-8 --
import sys

from Abyss.interaction.interaction import Interactor
from hand_key.models.resnet import resnet50
from hand_track.tracker import Tracker
from hand_detection.utils.datasets import LoadImages, LoadStreams
from hand_detection.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from hand_detection.utils.torch_utils import select_device, time_synchronized
import numpy as np
import argparse
import os
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, './hand_detection')

# os.environ['CUDA_ENABLE_DEVICES'] = '0'
# torch.cuda.set_device(0)


def detect(opt):
    # 检查参数--------------------------------------------------------------------------
    out, source, yolov5_weights, view, save = opt.output, opt.source, opt.yolov5_weights, opt.view, opt.save

    webcam = source == '0' or source.endswith('.txt') or source.startswith('rtsp') or source.startswith('http')

    pose_thres = opt.pose_thres

    res50_img_size, res50_weight = opt.res50_img_size, opt.res50_weight

    vid_path, vid_writer = None, None

    # 使用CUDA则启用半精度浮点数
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # 检查输出文件夹
    if not os.path.exists(out):
        os.makedirs(out)

    # 加载yolov5--------------------------------------------------------------------------
    model_yolo5 = torch.load(yolov5_weights, map_location=device)['model']
    model_yolo5.float().to(device).eval()
    stride = int(model_yolo5.stride.max())  # model stride
    yolov5_img_size = check_img_size(opt.yolov5_img_size, s=stride)  # check img_size
    if half:
        model_yolo5.half()  # to FP16
    names = model_yolo5.module.names if hasattr(model_yolo5, 'module') else model_yolo5.names  # 获取分类名
    print('load model : {}'.format(yolov5_weights))

    # 加载resnet50--------------------------------------------------------------------------
    model_res50 = resnet50(num_classes=42, img_size=res50_img_size[0])
    model_res50.to(device).eval()
    if half:
        model_res50.half()

    chkpt = torch.load(res50_weight, map_location=device)
    model_res50.load_state_dict(chkpt)
    print('load model : {}'.format(res50_weight))

    # 初始化追踪状态器--------------------------------------------------------------------------
    tracker = Tracker(opt.pose_cfg, pose_thres=pose_thres)

    # 初始化交互模块--------------------------------------------------------------------------
    interactor = Interactor()

    # Dataloader--------------------------------------------------------------------------
    if webcam:
        view = check_imshow()
        cudnn.benchmark = True  # 加快在视频中恒定大小图像的推断
        dataset = LoadStreams(source, img_size=yolov5_img_size, stride=stride)
    else:
        view = True
        dataset = LoadImages(source, img_size=yolov5_img_size, stride=stride)

    # 开始推理--------------------------------------------------------------------------------------------
    t0 = time.time()
    # 预热模型
    img = torch.zeros((1, 3, yolov5_img_size, yolov5_img_size), device=device)
    _ = model_yolo5(img.half() if half else img) if device.type != 'cpu' else None
    img = torch.zeros((1, 3, res50_img_size[0], res50_img_size[0]), device=device)
    _ = model_res50(img.half() if half else img) if device.type != 'cpu' else None

    # 对每张/帧图片进行处理。(识别文件的路径，yolo尺寸(3,h,w)，原始图片(h,w,3)，none)
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        # yolo部分图片预处理--------------------------------------------------------------------------
        img = torch.from_numpy(img).to(device)  # 将图片转成tensor并指认设备
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 加了一个维度

        # Inference
        t1 = time_synchronized()
        pred = model_yolo5(img, augment=opt.augment)[0]  # 得到预测结果列表

        # 非极大值抑制 最大检测数2 最小框边长：握拳的边长*0.9
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0])

        # 从pred列表中取出数据，如果是视频就只有一次循环，仅包含tensor[2,6]  2只手，6个信息
        for b, det_box in enumerate(pred):
            if webcam:  # batch_size >= 1
                p, s, im0 = path[b], '%g: ' % b, im0s[b].copy()
            else:
                p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]
            save_path = str(Path(out) / Path(p).name)

            if det_box is not None and len(det_box):  # 如果检测到目标，则跟踪两只手的状态，并显示跟踪效果
                # 将预测框从yolov5_img_size缩放回原始坐标
                det_box[:, :4] = scale_coords(img.shape[2:], det_box[:, :4], im0.shape).round()

                # 手检测计数
                s += '%g %ss ' % (len(det_box), names[0])

                # 对每只手进行关键点预测，并返回原始坐标
                keypoint_list = []
                for one_hand_box in det_box.data.cpu().numpy():
                    # resnet50图像预处理 -----------------------------------------------------------------
                    cut_img = im0[int(one_hand_box[1]):int(one_hand_box[3]),
                              int(one_hand_box[0]):int(one_hand_box[2])]  # 先切y轴，再切x轴

                    # cv2.imshow('test', cut_img)
                    # cv2.waitKey(0)

                    key_img = cv2.resize(cut_img, (res50_img_size[1], res50_img_size[0]),
                                         interpolation=cv2.INTER_CUBIC)  # 缩放成res50输入尺寸
                    key_img = (key_img.astype(np.float32) - 128.) / 256.
                    key_img = torch.from_numpy(key_img.transpose(2, 0, 1)).unsqueeze_(0)
                    if torch.cuda.is_available():
                        key_img = key_img.cuda()
                    key_img = key_img.half() if half else key_img.float()
                    # 模型推理
                    key_output = model_res50(key_img)
                    key_output = key_output.cpu().detach().numpy()
                    key_output = np.squeeze(key_output)  # 预测值域[0,1]
                    hand_ = []
                    for i in range(int(key_output.shape[0] / 2)):
                        x = (key_output[i * 2 + 0] * float(cut_img.shape[1])) + int(one_hand_box[0])
                        y = (key_output[i * 2 + 1] * float(cut_img.shape[0])) + int(one_hand_box[1])
                        hand_.append([x, y])
                    keypoint_list.append(hand_)

                # 处理结果：追踪&打印----------------------------------------------------------------------------
                tracker.update(det_box, keypoint_list)

            else:
                tracker.update_nodata([0, 1])  # 当因为运动太快而丢失检测时可以忽略

            tracker.plot(im0)

            # 这里判断进入其他功能
            interactor.interact(im0, tracker.get_order())

            # Print time (yolov5 + NMS + keypoint + track + draw + interact + ...)
            t2 = time_synchronized()
            print('%s (%.3fs)' % (s, t2 - t1))  # 时间

            # Stream results  是否在ui界面中展示
            if view is None:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save:
                if dataset.mode == 'images':
                    print('saving img!')
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save:
        print('Results saved to %s' % os.getcwd() + os.sep + out)

    print('Done. (%.3fs)' % (time.time() - t0))


def parse_argument():
    parser = argparse.ArgumentParser()
    # file/folder, 0-webcam。不支持图片格式              inference/input/test_video2.mp4
    parser.add_argument('--source', type=str, default='inference/input/piano.mp4', help='source')
    # 输出文件夹
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    # 输出视频格式
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # 是否展示结果
    parser.add_argument('--view', default=True, help='display results')
    # 是否保存视频
    parser.add_argument("--save", type=str, default=False, help='save results')
    # 是否使用显卡+半精度
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # yolov5模型路径           hand_v5s/best   best_YOLOv5l  best_yolo5s_half  yolov5l_best
    parser.add_argument('--yolov5_weights', type=str, default='inference/weights/hand_weight/best_YOLOv5l.pt')
    # yolov5输入尺寸
    parser.add_argument('--yolov5_img_size', type=int, default=640, help='inference size (pixels)')
    # yolov5推理时进行多尺度，翻转等操作(TTA)
    parser.add_argument('--augment', action='store_true', default=False, help='augmented inference')
    # nms置信度阈值
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    # nms的IOU阈值
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    # tracker二维角度约束阈值
    parser.add_argument("--pose_thres", type=float, default=0.4, help='pose angle threshold')
    # tracker手势字典设置文件
    parser.add_argument("--pose_cfg", type=str, default='inference/weights/cfg_pose.json', help='pose_cfg')
    # res50模型路径
    parser.add_argument('--res50_weight', type=str, default='inference/weights/pose_weight/resnet50_2021-418.pth',
                        help='res50_weight')
    # res50输入尺寸
    parser.add_argument('--res50_img_size', type=tuple, default=(256, 256), help='res50_img_size')

    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    with torch.no_grad():
        detect(parse_argument())
