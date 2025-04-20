import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Union
from torch.utils.data import Dataset


def normalize_linearly(x, x_min: float, x_max: float, y_min: float, y_max: float):
    slope = (y_min - y_max) / (x_min - x_max)
    offset = y_max - slope * x_max
    return slope * x + offset


@dataclass
class RolloutData:
    lat_accel: Union[np.array, None] = None
    steer: Union[np.array, None] = None
    v_ego: Union[np.array, None] = None

    def is_valid(self) -> bool:
        if any(
            getattr(self, field.name) is None
            for field in self.__dataclass_fields__.values()
        ):
            return False
        if not (len(self.lat_accel) == len(self.steer)):
            return False
        if not (len(self.steer) == len(self.v_ego)):
            return False
        return True

    def __len__(self):
        if self.is_valid():
            return len(self.lat_accel)
        return 0


class DynamicsDataset(Dataset):

    def __init__(self, history_size: int = 1):
        if history_size < 1:
            raise ValueError("History size must be larger than 1.")

        self.history_size = history_size
        self.data: List[RolloutData] = []
        self.rollout_lengths: List[int] = []
        self.cumulative_lengths: List[int] = []

    def __len__(self):
        return self.cumulative_lengths[-1]

    def global_to_local_index(self, global_idx):
        for file_idx, length in enumerate(self.cumulative_lengths):
            if global_idx < length:
                if file_idx == 0:
                    local_idx = global_idx
                else:
                    local_idx = global_idx - self.cumulative_lengths[file_idx - 1]
                return file_idx, local_idx
        raise IndexError("Index out of range")

    def __getitem__(self, idx):
        rollout_idx, local_idx = self.global_to_local_index(idx)
        rollout_data = self.data[rollout_idx]
        history_buffer_slice = slice(local_idx, local_idx + self.history_size)
        steer_history = rollout_data.steer[history_buffer_slice].astype(np.float32)
        lat_accel_history = rollout_data.lat_accel[history_buffer_slice].astype(
            np.float32
        )
        v_ego_history = rollout_data.v_ego[history_buffer_slice].astype(np.float32)
        v_ego_history = normalize_linearly(
            v_ego_history, x_min=0, x_max=55, y_min=-1, y_max=1
        )
        x = np.concatenate((steer_history, lat_accel_history))
        y = rollout_data.lat_accel[local_idx + self.history_size].astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

    def add_rollout_data(self, rollout_data: RolloutData):
        if not rollout_data.is_valid():
            raise ValueError("Invalid rollout data")
        self.data.append(rollout_data)
        self.rollout_lengths.append(len(rollout_data) - self.history_size)
        self.cumulative_lengths = np.cumsum(self.rollout_lengths)

    def get_raw_data(self):
        steer = []
        lat_accel = []
        for rollout_data in self.data:
            steer.extend(rollout_data.lat_accel.tolist())
            lat_accel.extend(rollout_data.steer.tolist())
        return steer, lat_accel
