# 2018-9-30
# Author: Wang Yongjie
# Email: yongjie.wang@ntu.edu.sg

import sys
import numpy as np
from subprocess import Popen, PIPE


class gpu_usage(object):
    def __init__(self):
        self.gpu_percent = 0.
        self.gpu_mem_avail = 0.
        self.gpu_mem_used = 0.
        self.gpu_mem_total = 0.
        self.gpu_info()

    def gpu_info(self):
        try:
            #p = Popen(["nvidia-smi","--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu", "--format=csv,noheader,nounits"], stdout=PIPE)
            p = Popen(["nvidia-smi","--query-gpu=utilization.gpu,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], stdout=PIPE)
            stdout, stderr = p.communicate()
        except:
            return []

        #output = stdout.decode('UTF-8')
        output = stdout.splitlines()
        string2int = lambda line : [[float(j) for j in i.decode('UTF-8').split(',')] for i in line] # 
        output = string2int(output) # convert string array to float
        array = np.array(output)
        usage = np.mean(array, 0)
        self.gpu_percent = usage[0].item()
        self.gpu_mem_total = usage[1].item()
        self.gpu_mem_used = usage[2].item()
        self.gpu_mem_avail = usage[3].item()
    
        self.gpu_single_info = output




'''
if __name__ == "__main__":
    test = gpu_usage()
    print(test.gpu_percent)
    print(test.gpu_mem_avail)
    print(test.gpu_mem_used)
    print(test.gpu_mem_total)
    print(test.gpu_single_info)

'''
