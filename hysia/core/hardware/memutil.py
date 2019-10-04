# 2018-9-30
# Author: Wang Yongjie
# Email: yongjie.wang@ntu.edu.sg

import os
import psutil

class memory_usage(object):
    def __init__(self):
        self.avail = 0
        self.used = 0
        self.total = 0
        self.GB = 1024 * 1024 * 1024
        self.memory_info()
        return

    def memory_info(self):
        tmp = psutil.virtual_memory()
        self.avail = tmp.available / self.GB
        self.used = tmp.used  / self.GB 
        self.total = tmp.total / self.GB
        return

'''
# local test module
if __name__ == "__main__":
    test = memory_usage()
    print(test.avail)
    print(test.used)
    print(test.total)
'''
