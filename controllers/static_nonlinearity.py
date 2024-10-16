import numpy as np
from abc import ABC, abstractmethod


class StaticNonlinearity(ABC):

    @abstractmethod
    def calculate_lataccel_from_torque(self, torque: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def calculate_torque_from_lataccel(self, lataccel: float) -> float:
        raise NotImplementedError


class ArctanNonlinearity(StaticNonlinearity):

    def __init__(self, A: float, B: float):
        self._A = A
        self._B = B

    def calculate_lataccel_from_torque(self, torque):
        lataccel = (1 / self._B) * np.tan(torque / self._A)
        return lataccel

    def calculate_torque_from_lataccel(self, lataccel: float) -> float:
        torque = self._A * np.arctan(self._B * lataccel)
        return torque
