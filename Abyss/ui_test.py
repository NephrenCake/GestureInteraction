# -- coding: utf-8 --
import sys
from threading import Thread

import cv2
import torch
from PySide2.QtWidgets import QMainWindow, QApplication
from Abyss.UI.main_ui import Ui_mainWindow
from Abyss.UI.ui_utils import update_video
from Abyss.inference import parse_argument, detect


class LoginGUI(QMainWindow, Ui_mainWindow):
    def __init__(self):
        super(LoginGUI, self).__init__()  # 调用父类
        self.setupUi(self)  # 初始化界面
        self.label.setScaledContents(True)  # 让视频流自适应 label 大小

        self.pushButton_4.clicked.connect(self.test_print)

    def test_print(self):
        print("test")

    def start_inference(self):
        """opt = parse_argument()
        opt.ui = self
        with torch.no_grad():
            detect(opt)"""

        with torch.no_grad():
            opt = parse_argument()
            opt.ui = self

            # 创建线程
            thread = Thread(target=detect,
                            args=(opt,)
                            )
            thread.start()


if __name__ == "__main__":
    app = QApplication([])
    gui = LoginGUI()  # 初始化
    gui.show()  # 将窗口控件显示在屏幕上
    gui.start_inference()
    sys.exit(app.exec_())
