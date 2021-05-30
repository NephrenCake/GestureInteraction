# -- coding: utf-8 --
import cv2


def compute_color_for_labels(label):
    color = (255, 0, 0) if label == 0 else (0, 0, 255)
    return tuple(color)


def draw(img, target, keypoint, offset=(0, 0)):
    # box text
    conf = target[4]
    id = int(target[5])
    iou = target[6]
    pose = target[7]
    content1 = 'id:' + str(id) + ' pose:' + str(pose)
    content2 = 'iou:' + "%.2f" % iou + ' conf:' + "%.2f" % conf
    # 画框
    draw_hand_box(img, int(target[0]), int(target[1]), int(target[2]), int(target[3]), id, content1, content2, offset)
    # 画线
    draw_hand_line(img, keypoint, offset)
    # 画点
    draw_hand_point(img, keypoint, offset)


def draw_hand_box(img_, x1, y1, x2, y2, id, content1=None, content2=None, offset=(0, 0)):
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]
    color = compute_color_for_labels(id)
    t_size = cv2.getTextSize(content1, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img_, (x1, y1), (x2, y2), (123, 104, 238), 2)
    cv2.putText(img_, content1, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 255], 2)
    cv2.putText(img_, content2, (x1, y1 + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 255], 2)
    # todo 美化视图


def draw_hand_line(img_, hand_, offset=(0, 0)):
    thick = 5  # 2
    colors = [(0, 215, 255), (255, 115, 55), (5, 255, 55), (25, 15, 255), (225, 15, 55), (255, 0, 255)]

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[1][0] + offset[0]), int(hand_[1][1] + offset[1])), colors[0], thick)
    cv2.line(img_, (int(hand_[1][0] + offset[0]), int(hand_[1][1] + offset[1])),
             (int(hand_[2][0] + offset[0]), int(hand_[2][1] + offset[1])), colors[0], thick)
    cv2.line(img_, (int(hand_[2][0] + offset[0]), int(hand_[2][1] + offset[1])),
             (int(hand_[3][0] + offset[0]), int(hand_[3][1] + offset[1])), colors[0], thick)
    cv2.line(img_, (int(hand_[3][0] + offset[0]), int(hand_[3][1] + offset[1])),
             (int(hand_[4][0] + offset[0]), int(hand_[4][1] + offset[1])), colors[0], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[5][0] + offset[0]), int(hand_[5][1] + offset[1])), colors[1], thick)
    cv2.line(img_, (int(hand_[5][0] + offset[0]), int(hand_[5][1] + offset[1])),
             (int(hand_[6][0] + offset[0]), int(hand_[6][1] + offset[1])), colors[1], thick)
    cv2.line(img_, (int(hand_[6][0] + offset[0]), int(hand_[6][1] + offset[1])),
             (int(hand_[7][0] + offset[0]), int(hand_[7][1] + offset[1])), colors[1], thick)
    cv2.line(img_, (int(hand_[7][0] + offset[0]), int(hand_[7][1] + offset[1])),
             (int(hand_[8][0] + offset[0]), int(hand_[8][1] + offset[1])), colors[1], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[9][0] + offset[0]), int(hand_[9][1] + offset[1])), colors[2], thick)
    cv2.line(img_, (int(hand_[9][0] + offset[0]), int(hand_[9][1] + offset[1])),
             (int(hand_[10][0] + offset[0]), int(hand_[10][1] + offset[1])), colors[2], thick)
    cv2.line(img_, (int(hand_[10][0] + offset[0]), int(hand_[10][1] + offset[1])),
             (int(hand_[11][0] + offset[0]), int(hand_[11][1] + offset[1])), colors[2], thick)
    cv2.line(img_, (int(hand_[11][0] + offset[0]), int(hand_[11][1] + offset[1])),
             (int(hand_[12][0] + offset[0]), int(hand_[12][1] + offset[1])), colors[2], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[13][0] + offset[0]), int(hand_[13][1] + offset[1])), colors[3], thick)
    cv2.line(img_, (int(hand_[13][0] + offset[0]), int(hand_[13][1] + offset[1])),
             (int(hand_[14][0] + offset[0]), int(hand_[14][1] + offset[1])), colors[3], thick)
    cv2.line(img_, (int(hand_[14][0] + offset[0]), int(hand_[14][1] + offset[1])),
             (int(hand_[15][0] + offset[0]), int(hand_[15][1] + offset[1])), colors[3], thick)
    cv2.line(img_, (int(hand_[15][0] + offset[0]), int(hand_[15][1] + offset[1])),
             (int(hand_[16][0] + offset[0]), int(hand_[16][1] + offset[1])), colors[3], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[17][0] + offset[0]), int(hand_[17][1] + offset[1])), colors[4], thick)
    cv2.line(img_, (int(hand_[17][0] + offset[0]), int(hand_[17][1] + offset[1])),
             (int(hand_[18][0] + offset[0]), int(hand_[18][1] + offset[1])), colors[4], thick)
    cv2.line(img_, (int(hand_[18][0] + offset[0]), int(hand_[18][1] + offset[1])),
             (int(hand_[19][0] + offset[0]), int(hand_[19][1] + offset[1])), colors[4], thick)
    cv2.line(img_, (int(hand_[19][0] + offset[0]), int(hand_[19][1] + offset[1])),
             (int(hand_[20][0] + offset[0]), int(hand_[20][1] + offset[1])), colors[4], thick)

    # -----------------------------------------------------------------------------二维角度演示
    """cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[1][0] + offset[0]), int(hand_[1][1] + offset[1])), colors[0], thick)
    cv2.line(img_, (int(hand_[3][0] + offset[0]), int(hand_[3][1] + offset[1])),
             (int(hand_[4][0] + offset[0]), int(hand_[4][1] + offset[1])), colors[0], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[5][0] + offset[0]), int(hand_[5][1] + offset[1])), colors[1], thick)
    cv2.line(img_, (int(hand_[6][0] + offset[0]), int(hand_[6][1] + offset[1])),
             (int(hand_[8][0] + offset[0]), int(hand_[8][1] + offset[1])), colors[1], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[9][0] + offset[0]), int(hand_[9][1] + offset[1])), colors[2], thick)
    cv2.line(img_, (int(hand_[10][0] + offset[0]), int(hand_[10][1] + offset[1])),
             (int(hand_[12][0] + offset[0]), int(hand_[12][1] + offset[1])), colors[2], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[13][0] + offset[0]), int(hand_[13][1] + offset[1])), colors[3], thick)
    cv2.line(img_, (int(hand_[14][0] + offset[0]), int(hand_[14][1] + offset[1])),
             (int(hand_[16][0] + offset[0]), int(hand_[16][1] + offset[1])), colors[3], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[17][0] + offset[0]), int(hand_[17][1] + offset[1])), colors[4], thick)
    cv2.line(img_, (int(hand_[18][0] + offset[0]), int(hand_[18][1] + offset[1])),
             (int(hand_[20][0] + offset[0]), int(hand_[20][1] + offset[1])), colors[4], thick)

    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[4][0] + offset[0]), int(hand_[4][1] + offset[1])), colors[5], thick)
    cv2.line(img_, (int(hand_[0][0] + offset[0]), int(hand_[0][1] + offset[1])),
             (int(hand_[8][0] + offset[0]), int(hand_[8][1] + offset[1])), colors[5], thick)"""


def draw_hand_point(img_, hand_, offset=(0, 0)):
    for i in range(21):
        x = int(hand_[i][0] + offset[0])
        y = int(hand_[i][1] + offset[1])

        cv2.circle(img_, (x, y), 3, (255, 50, 60), -1)  # 中心 2
        cv2.circle(img_, (x, y), 6, (123, 104, 238), 1)  # 中心 3

        # cv2.putText(img_, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 50, 60), 3)
