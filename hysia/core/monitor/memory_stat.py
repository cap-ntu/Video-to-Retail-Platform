# Desc: Memory statistics.
# Author: Zhou Shengsheng
# Date: 24/04/19


class MemoryStat(object):
    """
    Memory statistics (unit in bytes).

    Attributes:
        total (int): Total physical memory.
        available (int): The memory that can be given instantly to processes without the system going into swap.
        utilization (float): Memory utilization.
        free (int): Memory not being used at all (zeroed) that is readily available.
        used (int): Used memory.
        shared (int): memory that may be simultaneously accessed by multiple processes.
        buffered (int): Cache for things like file system metadata.
        cached (int): Cache for various things.
        totalSwap (int): Total swap memory.
        usedSwap (int): Used swap memory.
        freeSwap (int): Free swap memory.
        swapUtilization (int): Swap memory utilization.
    """

    def __init__(self, total, available, utilization, free, used, shared, buffered, cached, 
            totalSwap, usedSwap, freeSwap, swapUtilization):
        self.total = total
        self.available = available
        self.utilization = utilization
        self.free = free
        self.used = used
        self.shared = shared
        self.buffered = buffered
        self.cached = cached
        self.totalSwap = totalSwap
        self.usedSwap = usedSwap
        self.freeSwap = freeSwap
        self.swapUtilization = swapUtilization

    def __str__(self):
        return str(self.__dict__)

