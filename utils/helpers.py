from enum import Enum


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
