# Desc: Process statistics.
# Author: Zhou Shengsheng
# Date: 25/04/19

from enum import Enum
import psutil


class ProcessStatus(Enum):
    RUNNING = 0
    SLEEPING = 1
    DISK_SLEEP = 2
    STOPPED = 3
    TRACING_STOP = 4
    ZOMBIE = 5
    DEAD = 6
    WAKE_KILL = 7
    WAKING = 8
    PARKED = 9
    UNKNOWN = 10

    @classmethod
    def fromPsutil(cls, value):
        """Convert process status defined in psutil to this enum."""
        if value == psutil.STATUS_RUNNING:
            return cls.RUNNING
        if value == psutil.STATUS_SLEEPING:
            return cls.SLEEPING
        if value == psutil.STATUS_DISK_SLEEP:
            return cls.DISK_SLEEP
        if value == psutil.STATUS_STOPPED:
            return cls.STOPPED
        if value == psutil.STATUS_TRACING_STOP:
            return cls.TRACING_STOP
        if value == psutil.STATUS_ZOMBIE:
            return cls.ZOMBIE
        if value == psutil.STATUS_DEAD:
            return cls.DEAD
        if value == psutil.STATUS_WAKE_KILL:
            return cls.WAKE_KILL
        if value == psutil.STATUS_WAKING:
            return cls.WAKING
        if value == psutil.STATUS_PARKED:
            return cls.PARKED
        return cls.UNKNOWN


class ProcessStat(object):
    """
    Process statistics.

    Attributes:
        pid (int): Process id.
        name (str): Process name.
        ppid (int): Parent process id.
        cpuTimes (tuple): A (user, system, children_user, children_system) tuple 
            representing the accumulated process time, in seconds.
        cpuUtilization: CPU utilization.
        memoryInfo (tuple): A tuple representing memory info containing:
            * rss: Aka “Resident Set Size”, this is the non-swapped physical memory a process has used.
            * vms: Aka “Virtual Memory Size”, this is the total amount of virtual memory used by the process.
            * shared: memory that could be potentially shared with other processes.
            * text: Aka TRS (text resident set) the amount of memory devoted to executable code.
            * data: Aka DRS (data resident set) the amount of physical memory devoted to other than executable code.
            * lib: the memory used by shared libraries
            * dirty: the number of dirty pages.
        status (ProcessStatus): Process status.
        nice (int): Process niceness (priority, -20 to 20, lower the value higher the priority).
        ioNice (int): Process niceness for io operations (0 to 7, lower the value higher the priority).
        ctxSwitches (tuple): The number voluntary and involuntary context switches performed by this process (cumulative).
        fdCount (int): The number of file descriptors currently opened by this process (non cumulative).
        threadCount (int): The number of threads currently used by this process (non cumulative).
    """

    def __init__(self, pid, name, ppid, cpuTimes, cpuUtilization, memoryInfo, status,
            nice, ioNice, ctxSwitches, fdCount, threadCount):
        self.pid = pid
        self.name = name
        self.ppid = ppid
        self.cpuTimes = cpuTimes
        self.cpuUtilization = cpuUtilization
        self.memoryInfo = memoryInfo
        self.status = status
        self.nice = nice
        self.ioNice = ioNice
        self.ctxSwitches = ctxSwitches
        self.fdCount = fdCount
        self.threadCount = threadCount

    def __str__(self):
        return str(self.__dict__)

