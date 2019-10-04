# Desc: Monitor test script.
# Author: Zhou Shengsheng
# Date: 24/04/19
# Note: Please install modifed gputil: pip install git+git://github.com/ZhouShengsheng/gputil.git@master

import time
import os
import sys
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, ".."))
import _init_paths

from core.monitor import *


# Create monitor instance
monitor = Monitor()

# Get system stat
sysStat = monitor.getSysStat()
# print("System stat:", sysStat)


# Get individual stats
print("===> Get individual stats\n")

print("Uptime:", sysStat.upTime)
print()

print("CPU stat:", sysStat.cpuStat)
print()

print("Memory stat:", sysStat.memoryStat)
print()

print("GPU count:", sysStat.gpuCount)
print("GPU stats:")
for gpuStat in sysStat.gpuStats:
    print(gpuStat)
print()

print("Current process stat:", sysStat.processStat)
print("Process count:", sysStat.processCount)
# Too many processes
# print("Process stats:")
# for processStat in sysStat.processStats:
#     print(processStat)
print("5 example process stats:")
for i in range(5):
    print(sysStat.processStats[i])
print()

print("GPU process count:", sysStat.gpuProcessCount)
print("GPU process stats:")
for gpuProcessStat in sysStat.gpuProcessStats:
    print(gpuProcessStat)
print()

print("Network count:", sysStat.networkCount)
print("Network stats:")
for networkStat in sysStat.networkStats:
    print(networkStat)
print()


# Performance testing
print("===> Performance testing\n")

# Test create monitor
print("Test create monitor...")
tick = time.time()
monitor = Monitor()
print("Done. Time cost:", time.time() - tick, "\n")

# Test get all stat
print("Test get sys stat...")
tick = time.time()
monitor.getSysStat()
print("Done. Time cost:", time.time() - tick, "\n")

# Test get cpu stat
print("Test get cpu stat...")
tick = time.time()
monitor._Monitor__queryCPUStat()  # Call private method
print("Done. Time cost:", time.time() - tick, "\n")

# Test get gpu stats
print("Test get gpu stats...")
tick = time.time()
monitor._Monitor__queryGPUStats()
print("Done. Time cost:", time.time() - tick, "\n")

# Test get process stats
print("Test get process stats...")
tick = time.time()
monitor._Monitor__queryGPUStats()
print("Done. Time cost:", time.time() - tick, "\n")

# Test get gpu process stats
print("Test get gpu process stats...")
tick = time.time()
monitor._Monitor__queryProcessStats()
print("Done. Time cost:", time.time() - tick, "\n")

# Test get network stats
print("Test get network stats...")
tick = time.time()
monitor._Monitor__queryNetworkStats()
print("Done. Time cost:", time.time() - tick, "\n")

