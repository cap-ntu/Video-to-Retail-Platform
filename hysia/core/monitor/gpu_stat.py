# Desc: GPU statistics.
# Author: Zhou Shengsheng
# Date: 24/04/19

class GPUStat(object):
    """
    GPU statistics.

    Attributes:
        gpuId (int): GPU id or index.
        uuid (str): GPU uuid.
        model (str): Manufacturing model.
        totalMemory (float): Total memory in MB.
        usedMemory (float): Used memory in MB.
        freeMemory (float): Free memory in MB.
        memoryUtilization (float): Memory utilization.
        temperature (float): Temerature in celsius degree.
    """

    def __init__(self, gpuId, uuid, model, totalMemory, usedMemory, freeMemory, memoryUtilization, temperature):
        self.gpuId = gpuId
        self.uuid = uuid
        self.model = model
        self.totalMemory = totalMemory
        self.usedMemory = usedMemory
        self.freeMemory = freeMemory
        self.memoryUtilization = memoryUtilization
        self.temperature = temperature

    def __str__(self):
        return str(self.__dict__)

