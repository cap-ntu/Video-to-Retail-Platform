# 2018-9-30
# Author: Wang Yongjie
# Email: yongjie.wang@ntu.edu.sg

import os
import psutil

class cpu_usage(object):
    def __init__(self):
        self.cpu_num = 0
        self.cpu_frequency = 0
        self.cpu_percent = 0
        self.cpu_info()

    def cpu_info(self):
        self.cpu_num = psutil.cpu_count()
        self.cpu_frequency = psutil.cpu_freq().current #pstuil.cpu_freq() is a namedtuple object
        self.cpu_percent = psutil.cpu_percent()
'''
if __name__ == "__main__":
    test = cpu_usage()
    print(test.cpu_num)
    print(test.cpu_frequency)
    print(test.cpu_percent)
'''
