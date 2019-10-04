# Desc: System statistics definition.
# Author: Zhou Shengsheng
# Date: 24/04/19


class SysStat(object):
    """
    System statistics including cpu, gpu, memory, processes, gpu processes and others.

    Attributes:
        upTime (int): Seconds from system boot.
        cpuStat (CPUStat): CPU statistics.
        memoryStat (MemoryStat): Memory statistics.
        gpuCount (int): GPU count.
        gpuStats (list): Stats for all GPUs.
        processStat (ProcessStat): Stat for the current process (self).
        processCount (int): Total process count in the system.
        processStats (list): Stats for all processes.
        gpuProcessCount (int): Count of processes using gpu.
        gpuProcessStats (list): Stats for all processes using gpu.
        networkCount (int): Network interface count.
        networkStats (list): Stats for all network interfaces.
    """

    def __init__(self):
        self.uptime = None
        self.cpuStat = None
        self.memoryStat = None
        self.gpuCount = None
        self.gpuStats = None
        self.processStat = None
        self.processCount = None
        self.processStats = None
        self.gpuProcessCount = None
        self.gpuProcessStats = None
        self.networkCount = None
        self.networkStats = None

    def __str__(self):
        return str(self.__dict__)

