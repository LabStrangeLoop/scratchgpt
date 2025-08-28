from abc import ABC, abstractmethod

import numpy as np


class Meter(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def add(self, value: float, n: int = 1) -> None:
        pass

    @abstractmethod
    def value(self) -> tuple[float, float]:
        pass


class AverageValueMeter(Meter):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.val: float = 0.0

    def add(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self) -> tuple[float, float]:
        return self.mean, self.std

    def get_name(self) -> str:
        return __name__

    def reset(self) -> None:
        self.n: int = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
