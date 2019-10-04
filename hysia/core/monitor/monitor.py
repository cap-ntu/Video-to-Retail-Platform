# Desc: Monitor to keep track of system statistics including CPU, GPU, memory, process and network.
# Author: Zhou Shengsheng
# Date: 24/04/19
# References:
#   (1) Get cpu info:
#         * py-cpuinfo: https://github.com/workhorsy/py-cpuinfo
#         * psutil: https://github.com/giampaolo/psutil
#   (2) Get gpu info:
#         * gputil: https://github.com/ZhouShengsheng/gputil
#           install cmd: pip install git+git://github.com/ZhouShengsheng/gputil.git@master 
#   (3) Get memory, process and network stats:
#         * psutil: https://github.com/giampaolo/psutil

import time
import os

import cpuinfo
import psutil
import GPUtil

from .sys_stat import SysStat
from .cpu_stat import CPUStat
from .memory_stat import MemoryStat
from .gpu_stat import GPUStat
from .process_stat import *
from .gpu_process_stat import GPUProcessStat
from .network_stat import *


class Monitor(object):
    """Monitor to keep track of system statistics including CPU, GPU, memory, process and network."""

    def __init__(self):
        # Query and cache cpu static stat
        self.__cachedCPUStaticStat = self.__queryCPUStaticStat()
        # Create sys stat
        self.__sysStat = SysStat()

    def getSysStat(self):
        """
        Get system statistics. This function will always get the latest system stats.
        
        Returns:
            sysStat (SysStat): System statistics.
        """
        sysStat = self.__sysStat
        sysStat.upTime = time.time() - psutil.boot_time()
        sysStat.cpuStat = self.__queryCPUStat()
        sysStat.memoryStat = self.__queryMemoryStat()
        sysStat.gpuStats = self.__queryGPUStats()
        sysStat.gpuCount = len(sysStat.gpuStats)
        sysStat.processStat, sysStat.processStats = self.__queryProcessStats()
        sysStat.processCount = len(sysStat.processStats)
        sysStat.gpuProcessStats = self.__queryGPUProcessStats()
        sysStat.gpuProcessCount = len(sysStat.gpuProcessStats)
        sysStat.networkStats = self.__queryNetworkStats()
        sysStat.networkCount = len(sysStat.networkStats)

        return self.__sysStat

    def __queryCPUStaticStat(self):
        """
        Query cpu static stat.

        Returns:
            cpuStaticStat (list): CPU static statistics including 
                model, count, freqs and cache.
        """
        cpuInfo = cpuinfo.get_cpu_info()
        model = cpuInfo['brand']
        count = cpuInfo['count']
        extractFloat = lambda s: float(s.split()[0])
        cache = (extractFloat(cpuInfo['l1_data_cache_size']) + 
                 extractFloat(cpuInfo['l1_instruction_cache_size']) +
                 extractFloat(cpuInfo['l2_cache_size']) +
                 extractFloat(cpuInfo['l3_cache_size']))
        freqs = psutil.cpu_freq()
        freqs = (freqs[0], freqs[1], freqs[2])
        
        return (model, count, freqs, cache)

    def __queryCPUStat(self):
        """
        Query cpu stat.

        Returns:
            cpuStat (CPUStat): CPU statistics.
        """
        cpuStaticStat = self.__cachedCPUStaticStat
        loads = os.getloadavg()
        utilization = psutil.cpu_percent() / 100.
        cpuTimes = tuple(psutil.cpu_times())
        cpuTimesRatio = tuple(x / 100. for x in psutil.cpu_times_percent())
        return CPUStat(cpuStaticStat[0], cpuStaticStat[1], cpuStaticStat[2], cpuStaticStat[3],
                    loads, utilization, cpuTimes, cpuTimesRatio)

    def __queryMemoryStat(self):
        """
        Query memory stat.

        Returns:
            memoryStat (MemoryStat): Memory statistics.
        """
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return MemoryStat(vm[0], vm[1], vm[2] / 100., vm[3], vm[4], vm[7], vm[8], vm[9],
                swap[0], swap[1], swap[2], swap[3] / 100.)

    def __queryGPUStats(self):
        """
        Query stats for all GPUs.

        Returns:
            gpuStats (list): GPU statistics.
        """
        gpus = GPUtil.getGPUs()
        if gpus:
            return [GPUStat(gpu.id, gpu.uuid, gpu.name, gpu.memoryTotal, gpu.memoryUsed, 
                gpu.memoryFree, gpu.memoryUtil, gpu.temperature) for gpu in gpus]
        return []

    def __queryProcessStats(self):
        """
        Query stats for all processes.

        Returns:
            processStats(list): Process statistics.
        """
        selfStat = None
        stats = []
        # Get current pid
        pid = os.getpid()
        # Iterate over all processes
        for proc in psutil.process_iter():
            try:
                pinfo = proc.as_dict()
            except psutil.NoSuchProcess:
                pass
            else:
                cpuTimes = pinfo['cpu_times']
                cpuTimes = (cpuTimes.user, cpuTimes.system, 
                        cpuTimes.children_user, cpuTimes.children_system)
                memoryInfo = pinfo['memory_info']
                memoryInfo = (memoryInfo.rss, memoryInfo.vms, memoryInfo.shared,
                        memoryInfo.text, memoryInfo.lib, memoryInfo.data, memoryInfo.dirty)
                status = ProcessStatus.fromPsutil(proc.status())
                ctxSwitches = (pinfo['num_ctx_switches'].voluntary, pinfo['num_ctx_switches'].involuntary)
                fdCount = pinfo['num_fds'] if pinfo['num_fds'] else 0
                threadCount = pinfo['num_threads'] if pinfo['num_threads'] else 0
                stat = ProcessStat(pinfo['pid'], pinfo['name'], pinfo['ppid'], cpuTimes, 
                        pinfo['cpu_percent'] / 100., memoryInfo, status, pinfo['nice'], 
                        pinfo['ionice'].value, ctxSwitches, fdCount, threadCount)
                if not selfStat and pid == stat.pid:
                    selfStat = stat

                stats.append(stat)
        return selfStat, stats

    def __queryGPUProcessStats(self):
        """
        Query stats for all GPU processes.

        Returns:
            gpuProcessStats (list): GPU process statistics.
        """
        processes = GPUtil.getGPUProcesses()
        if processes:
            return [GPUProcessStat(proc.pid, proc.processName, proc.gpuId, proc.gpuUuid,
                proc.usedMemory) for proc in processes]
        return []

    def __queryNetworkStats(self):
        ifStatDict = psutil.net_if_stats()
        if not ifStatDict:
            return []
        ifAddrDict = psutil.net_if_addrs()
        stats = []
        for nic, ifStat in ifStatDict.items():
            stat = NetworkStat()
            stat.nic = nic
            stat.isUp = ifStat.isup
            stat.duplex = NicDuplexType.fromPsutil(ifStat.duplex)
            stat.speed = ifStat.speed
            stat.mtu = ifStat.mtu
            ifAddrs = ifAddrDict[nic]
            addrs = []
            for ifAddr in ifAddrs:
                addrs.append((ifAddr.family, ifAddr.address, ifAddr.netmask,
                    ifAddr.broadcast, ifAddr.ptp))
            stat.addrs = addrs
            stats.append(stat)
        return stats

