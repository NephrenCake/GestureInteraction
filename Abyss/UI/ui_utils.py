# -- coding: utf-8 --
import cv2
from PySide2.QtGui import QImage, QPixmap


def update_video(ui, frame):
    # Mat格式图像转Qt中图像的方法     读取了一张图片并展示
    show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
    ui.label.setPixmap(QPixmap.fromImage(showImage))
    # cv2.waitKey(33)
    # if not self.window().isVisible():
    #     return
