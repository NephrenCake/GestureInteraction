# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_ui.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        if not mainWindow.objectName():
            mainWindow.setObjectName(u"mainWindow")
        mainWindow.resize(1280, 720)
        mainWindow.setMinimumSize(QSize(1280, 720))
        mainWindow.setMaximumSize(QSize(1280, 720))
        mainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(mainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(0, 0, 1280, 720))
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(0, 0, 1280, 720))
        self.line = QFrame(self.groupBox)
        self.line.setObjectName(u"line")
        self.line.setWindowModality(Qt.NonModal)
        self.line.setGeometry(QRect(190, 640, 900, 16))
        self.line.setStyleSheet(u"")
        self.line.setFrameShadow(QFrame.Sunken)
        self.line.setFrameShape(QFrame.HLine)
        self.groupBox_2 = QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(190, 550, 900, 60))
        self.groupBox_2.setStyleSheet(u"QPushButton { \n"
                                      "    background: rgba(0, 255, 255, 0.1);\n"
                                      "    color: rgba(148, 0, 211, 0.5) ;\n"
                                      "    border: none;\n"
                                      "    font-size:18px;\n"
                                      "}\n"
                                      "\n"
                                      "QPushButton:disabled {  \n"
                                      "    background: rgba(255, 0, 255, 0.1);\n"
                                      "    color: red;\n"
                                      "}\n"
                                      "\n"
                                      "QGroupBox { \n"
                                      "    background: rgba(255, 255, 255, 0);\n"
                                      "    border: none;\n"
                                      "}")
        self.pushButton_1 = QPushButton(self.groupBox_2)
        self.pushButton_1.setObjectName(u"pushButton_1")
        self.pushButton_1.setEnabled(True)
        self.pushButton_1.setGeometry(QRect(240, 0, 60, 60))
        self.pushButton_2 = QPushButton(self.groupBox_2)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(420, 0, 60, 60))
        self.pushButton_3 = QPushButton(self.groupBox_2)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(600, 0, 60, 60))
        self.groupBox_3 = QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(190, 610, 900, 30))
        self.groupBox_3.setStyleSheet(u"QPushButton { \n"
                                      "    background: rgba(255, 255, 255, 0);\n"
                                      "    color: rgba(148, 0, 211, 0.8) ;\n"
                                      "    border: none;\n"
                                      "    font-size:18px;\n"
                                      "}\n"
                                      "\n"
                                      "QGroupBox { \n"
                                      "    background: rgba(255, 255, 255, 0);\n"
                                      "    border: none;\n"
                                      "}")
        self.pushButton_5 = QPushButton(self.groupBox_3)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setEnabled(True)
        self.pushButton_5.setGeometry(QRect(210, 0, 120, 30))
        self.pushButton_6 = QPushButton(self.groupBox_3)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(390, 0, 120, 30))
        self.pushButton_7 = QPushButton(self.groupBox_3)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(570, 0, 120, 30))
        self.pushButton_4 = QPushButton(self.groupBox)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setEnabled(True)
        self.pushButton_4.setGeometry(QRect(400, 160, 391, 161))
        self.label.raise_()
        self.groupBox_2.raise_()
        self.groupBox_3.raise_()
        self.line.raise_()
        self.pushButton_4.raise_()
        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)

        QMetaObject.connectSlotsByName(mainWindow)

    # setupUi

    def retranslateUi(self, mainWindow):
        mainWindow.setWindowTitle(QCoreApplication.translate("mainWindow", u"Abyss", None))
        self.groupBox.setTitle(QCoreApplication.translate("mainWindow", u"\u89c6\u9891\u6d41", None))
        self.label.setText("")
        self.groupBox_2.setTitle("")
        self.pushButton_1.setText(QCoreApplication.translate("mainWindow", u"B1", None))
        self.pushButton_2.setText(QCoreApplication.translate("mainWindow", u"B2", None))
        self.pushButton_3.setText(QCoreApplication.translate("mainWindow", u"B3", None))
        self.groupBox_3.setTitle("")
        self.pushButton_5.setText(QCoreApplication.translate("mainWindow", u"B1", None))
        self.pushButton_6.setText(QCoreApplication.translate("mainWindow", u"B2", None))
        self.pushButton_7.setText(QCoreApplication.translate("mainWindow", u"B3", None))
        self.pushButton_4.setText(QCoreApplication.translate("mainWindow", u"test", None))
    # retranslateUi
