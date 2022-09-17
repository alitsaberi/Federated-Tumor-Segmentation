from enum import Enum

import numpy as np


class EnumBase(Enum):
    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.lower()

    @classmethod
    def all(cls):
        return [str(p) for p in cls]

    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
