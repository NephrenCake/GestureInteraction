# -- coding: utf-8 --
import json
import os
import time

import torch
from playsound import playsound
from torchvision import transforms
from PIL import Image

from Abyss.interaction.resnet_classfication.model import resnet34


def flower_classfication(app_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    json_path = 'interaction/resnet_classfication/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "inference/weights/object_weight/resnet34-best.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()

    while True:
        time.sleep(0.01)

        # todo 优化清除逻辑

        if app_dict["img"] is not None and not app_dict["result"]:  # 没有需要预测的图像，并且不持有上一次检测的记录
            try:
                assert app_dict["left_top"] is not None and app_dict["right_bottom"] is not None

                # todo 预测done
                img = app_dict["img"].copy()
                img = img[int(app_dict["left_top"][1]):int(app_dict["right_bottom"][1]),
                      int(app_dict["left_top"][0]):int(app_dict["right_bottom"][0])]  # 先切y轴，再切x轴

                # print(img.shape)
                # print(app_dict["left_top"], app_dict["right_bottom"])

                img = Image.fromarray(img)  # ndarray->pil
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)
                with torch.no_grad():
                    # predict class
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                result = class_indict[str(predict_cla)]
                score = "{:.2}".format(predict[predict_cla].numpy())

                print("classfication success. {} with {:.3}".format(result, score))

                app_dict["result"] = {
                    "class": result,
                    "score": score
                }
                app_dict["img"] = None
                # todo 语音提示
                try:
                    playsound('inference/audio/' + result + '.mp3')
                except Exception as e:
                    print(e)

            except Exception as e:
                print(e)
                try:
                    playsound('inference/audio/' + '' + '.mp3')  # todo 识别失败语音
                except Exception as e:
                    print(e)

        else:
            if app_dict["mouse"] is not None:
                x = app_dict["mouse"][0]
                y = app_dict["mouse"][1]

                # print(app_dict["mouse"])
                # print(app_dict["left_top"], app_dict["right_bottom"])

                if app_dict["left_top"] is not None and app_dict["right_bottom"] is not None:
                    if x < app_dict["left_top"][0]:
                        app_dict["left_top"][0] = x
                        print("t1")
                    if y < app_dict["left_top"][1]:
                        app_dict["left_top"][1] = y
                        print("t2")
                    if x > app_dict["right_bottom"][0]:
                        app_dict["right_bottom"][0] = x
                        print("t3")
                    if y > app_dict["right_bottom"][1]:
                        app_dict["right_bottom"][1] = y
                        print("t4")

                    # print(app_dict["left_top"], app_dict["right_bottom"])

                else:
                    app_dict["left_top"] = [x, y]
                    app_dict["right_bottom"] = [x, y]

                app_dict["mouse"] = None
            else:
                continue
