# -*-coding:utf-8-*-

import argparse
import torch

from Abyss.hand_track.angle_util import pose_to_angles
from Abyss.hand_track.draw_util import draw_hand_line, draw_hand_point
from hand_key.models.resnet import resnet50
from hand_key.utils.common_utils import *


def main(choice):
    if choice == 1:
        # 构建模型----------------------------------------------------------------
        model_ = resnet50(num_classes=42, img_size=ops.img_size[0])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_.to(device).eval()  # 设置为前向推断模式

        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))

        # ---------------------------------------------------------------- 预测图片
        with torch.no_grad():
            idx = 0
            data_list = []
            for file in os.listdir(ops.test_path):
                if '.jpg' not in file:
                    continue
                idx += 1
                print('{}) image : {}'.format(idx, file))
                # img = cv2.imread(ops.test_path + file)  # img = None
                img = cv2.imdecode(np.fromfile(os.path.join(ops.test_path, file), dtype=np.uint8), -1)
                img_width = img.shape[1]
                img_height = img.shape[0]
                # 输入图片预处理
                img_ = cv2.resize(img, (ops.img_size[1], ops.img_size[0]), interpolation=cv2.INTER_CUBIC)
                img_ = img_.astype(np.float32)
                img_ = (img_ - 128.) / 256.
                img_ = img_.transpose(2, 0, 1)
                img_ = torch.from_numpy(img_).unsqueeze_(0)
                if torch.cuda.is_available():
                    img_ = img_.cuda()  # (bs, 3, h, w)

                # 模型推理
                pre_ = model_(img_.float())
                output = pre_.cpu().detach().numpy()
                output = np.squeeze(output)

                pts_hand = []  # 构建关键点结构
                for i in range(int(output.shape[0] / 2)):
                    x = (output[i * 2 + 0] * float(img_width))
                    y = (output[i * 2 + 1] * float(img_height))
                    pts_hand.append([x, y])

                if ops.vis:
                    draw_hand_line(img, pts_hand)
                    draw_hand_point(img, pts_hand)
                    cv2.imshow(file, img)
                    cv2.waitKey(0)

                # 计算角度
                angle = pose_to_angles(pts_hand)
                name = file.replace('.jpg', '', 1)
                data = {
                    'name': name,
                    'angle': angle.tolist()
                }
                data_list.append(data)
                print(data)
                # 保存图片
                cv2.imwrite("inference/output/"+name+".jpg", img)


            if ops.cfg_write:
                with open(ops.cfg_pose, 'w') as f:
                    json.dump(data_list, f)

        cv2.destroyAllWindows()

        print('well done ')
    elif choice == 2:
        if os.path.isfile(ops.cfg_pose):
            with open(ops.cfg_pose, 'r') as f:
                data = json.load(f)
            print(data)
            print(type(data))
        else:
            Exception('no data')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Hand Pose Setting')
    # 模型路径
    parser.add_argument('--model_path', type=str, default='inference/weights/pose_weight/resnet50_2021-418.pth', help='model_path')
    # GPU选择
    parser.add_argument('--GPUS', type=str, default='0', help='GPUS')
    # 设置手势图片路径  hand_key/image
    parser.add_argument('--test_path', type=str, default='inference/input/pose_setting', help='test_path')
    # 输入模型图片尺寸
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='img_size')
    # 是否可视化图片
    parser.add_argument('--vis', type=bool, default=True, help='vis')
    # 手势字典文件路径
    parser.add_argument('--cfg_pose', type=str, default='inference/weights/cfg_pose.json')
    # 重写设置文件
    parser.add_argument('--cfg_write', type=bool, default=True)

    ops = parser.parse_args()

    choice = input("1.设定手势\n2.查看手势\n")
    main(int(choice))
