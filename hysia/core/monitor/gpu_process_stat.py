# Desc: GPU process statistics.
# Author: Zhou Shengsheng
# Date: 25/04/19

class GPUProcessStat(object):
    """
    GPU process statistics.

    Attributes:
        pid (int): Process id.
        name (str): Process name.
        gpuId (int): GPU id or index.
        gpuUuid (str): GPU uuid.
        usedMemory (float): Used memory in MB for this process.
    """

    def __init__(self, pid, name, gpuId, gpuUuid, usedMemory):
        self.pid = pid
        self.name = name
        self.gpuId = gpuId
        self.gpuUuid = gpuUuid
        self.usedMemory = usedMemory

    def __str__(self):
        return str(self.__dict__)

