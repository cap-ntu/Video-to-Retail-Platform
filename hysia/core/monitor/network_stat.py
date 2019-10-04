# Desc: Network statistics.
# Author: Zhou Shengsheng
# Date: 25/04/19

from enum import Enum
import psutil


class NicDuplexType(Enum):
    """Duplex type for network interface card."""

    FULL = 0
    HALF = 1
    UNKNOWN = 2

    @classmethod
    def fromPsutil(cls, value):
        """Convert duplex type defined in psutil to this enum."""
        if value == psutil.NIC_DUPLEX_FULL:
            return cls.FULL
        if value == psutil.NIC_DUPLEX_HALF:
            return cls.HALF
        return cls.UNKNOWN


class NetworkStat(object):
    """
    Network statistics for a single nic.

    Attributes:
        nic (str): Network interface card identifier (e.g.: eth0).
        isUp (boolean): Is the nic up and running.
        duplex (NicDuplexType): The duplex communication type.
        speed (int): The NIC speed expressed in mega bits (MB), 
            if it can’t be determined (e.g. ‘localhost’) it will be set to 0.
        mtu (int): NIC’s maximum transmission unit expressed in bytes.
        addrs (tuple): A tuple containing addresses associated with this nic.
            Each address is a tuple containing family (socket.AddressFamily), 
            address (str), netmask (str), broadcast (str) and ptp (str).
    """

    def __init__(self):
        self.nic = None
        self.isUp = None
        self.duplex = None
        self.speed = None
        self.mtu = None
        self.addrs = None

    def __str__(self):
        return str(self.__dict__)

