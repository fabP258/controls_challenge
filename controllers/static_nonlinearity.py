import numpy as np


class StaticNonlinearity:

    def calculate_lataccel_from_torque(self, torque: float) -> float:
        pass

    def calculate_torque_from_lataccel(self, lataccel: float) -> float:
        pass


class ArctanNonlinearity(StaticNonlinearity):

    def __init__(self, A: float, B: float):
        self._A = A
        self._B = B

    def calculate_torque_from_lataccel(self, lataccel: float) -> float:
        torque = self._A * np.arctan(self._B * lataccel)
        return torque
