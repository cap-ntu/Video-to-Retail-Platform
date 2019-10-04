# Desc: CPU statistics.
# Author: Zhou Shengsheng
# Date: 24/04/19

class CPUStat(object):
    """
    CPU statistics.

    Attributes:
        model (str): Manufacturing model.
        count (int): Logical CPU count.
        freqs (tuple): Tuple containing current, min and max frequencies in Mhz.
        cache (int): Total cache in KB of L1, L2 and L3 caches.
        loads (list): The average system load over the last 1, 5 and 15 minutes.
        utilization (float): Current cpu utilization.
        times (tuple): system CPU times as a named tuple including user, system, idle,
            nice, iowait, irq, softirq, steal, guest, and guest_nice.
        timesRatio (tuple): Utilization for each specific cpu time.
    """

    def __init__(self, model, count, freqs, cache, loads, utilization, times, timesRatio):
        self.model = model
        self.count = count
        self.freqs = freqs
        self.cache = cache
        self.loads = loads
        self.utilization = utilization
        self.times = times
        self.timesRatio = timesRatio

    def __str__(self):
        return str(self.__dict__)

