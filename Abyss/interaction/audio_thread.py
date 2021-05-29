# -- coding: utf-8 --
import threading
import time

from playsound import playsound


# class myThread(threading.Thread):
#     def __init__(self, threadID, name, counter):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#
#     def run(self):
#         print("开始线程：" + self.name)
#         print_time(self.name, self.counter, 5)
#         print("退出线程：" + self.name)
#
#
# def print_time(threadName, delay, counter):
#     while counter:
#         if exitFlag:
#             threadName.exit()
#         time.sleep(delay)
#         print("%s: %s" % (threadName, time.ctime(time.time())))
#         counter -= 1


def play_sounds(act):
    temp = []
    while True:
        time.sleep(0.01)
        if len(act) != 0:
            for a in act:
                temp.append(a)
            act.clear()
        for t in temp:
            try:
                playsound('inference/audio/' + t + '.mp3')
            except Exception as e:
                print(e)
            temp.clear()


def play_piano(data):
    temp = []
    while True:
        time.sleep(0.01)
        if len(data) != 0:
            for a in data:
                if a not in temp:
                    temp.append(a)
            data.clear()
        for t in temp:
            try:
                playsound('inference/audio/' + t + '.mp3')
            except Exception as e:
                print(e)
            temp.clear()
