import numpy as np
import torch
from . import BaseController
import timeit
from collections import deque


class Controller(BaseController):

    def __init__(self, models: list):
        self.models = models
        self.history_size = 1
        self.T = 10
        self.num_particles = 20
        self.num_action_samples = 25
        self.num_elites = 5
        self.action_reward_weight = 0.1
        self.num_cem_updates = 5

        # cross entropy distribution
        self.action_dist_mu = 0.0 * np.ones(((self.T),))
        self.action_dist_cov = np.eye((self.T)) * 0.3

        # history buffers
        self.lat_accel_history = deque(
            [0.0 for _ in range(self.history_size)], maxlen=self.history_size
        )
        self.action_history = deque(
            [0.0 for _ in range(self.history_size - 1)], maxlen=self.history_size - 1
        )

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # TODO: remove this after initial trials
        if len(self.lat_accel_history) != self.history_size:
            raise ValueError("Size mismatch of state history")
        if len(self.action_history) != (self.history_size - 1):
            raise ValueError("Size mismatch of action history")

        start = timeit.default_timer()

        lat_accel_setpoints = target_lataccel * torch.ones(
            (self.T + 1)
        )  # TODO: use future_plan
        if len(lat_accel_setpoints) != (self.T + 1):
            raise ValueError("Size mismatch of state setpoints")

        self.lat_accel_history.popleft()
        self.lat_accel_history.append(current_lataccel)

        for j in range(self.num_cem_updates):
            # sample action sequence
            action_sequences = self.sample_action_sequences()
            action_sequences = torch.tensor(action_sequences)
            action_rewards = np.zeros((len(action_sequences),))
            for i, action_sequence in enumerate(action_sequences):
                action_rewards[i] = self.sample_trajectory_parallel(
                    action_sequence, lat_accel_setpoints
                )
            if j == (self.num_cem_updates - 1):
                best_action_idx = np.argmax(action_rewards)
                best_action_sequence = action_sequences[best_action_idx]
                action = best_action_sequence[0]
                if self.history_size > 1:
                    self.action_history.popleft()
                    self.action_history.append(action)
                end = timeit.default_timer()
                print(f"PETS-MPC took: {end - start} s")
                return action

            # update CEM distribution
            elite_indices = action_rewards.argsort()[-self.num_elites :]
            elite_actions = action_sequences[elite_indices]
            self.action_dist_mu = elite_actions.mean(axis=0)
            # self.action_dist_cov = np.cov(elite_actions.T) + 1e-6 * np.eye(elite_actions.shape[1])

    def sample_trajectory_parallel(
        self, action_sequence: np.array, lat_accel_setpoints: torch.tensor
    ) -> float:
        state_trajectories = torch.zeros(
            (self.num_particles, self.T + self.history_size)
        )
        state_trajectories[:, : self.history_size] = torch.tensor(
            self.lat_accel_history
        )
        action_sequence = torch.concatenate(
            (torch.tensor(self.action_history), action_sequence)
        )
        bootstrap_indices = np.random.randint(
            low=0, high=len(self.models), size=self.num_particles
        )
        x = torch.zeros((1, 2 * self.history_size), dtype=torch.float32)
        for p in range(self.num_particles):
            for t in range(self.T):
                x[:, : self.history_size] = action_sequence[t : t + self.history_size]
                x[:, self.history_size :] = state_trajectories[
                    p, t : t + self.history_size
                ]
                pred_state_mean, pred_state_log_var = self.models[bootstrap_indices[p]](
                    x
                )
                std = torch.exp(0.5 * pred_state_log_var)
                eps = torch.randn_like(std)
                sampled_state = pred_state_mean + eps * std
                state_trajectories[p, t + self.history_size] = sampled_state.squeeze()
        particle_rewards = -(
            (state_trajectories[:, self.history_size - 1 :] - lat_accel_setpoints) ** 2
        )
        particle_rewards[:, : -self.history_size] -= (
            self.action_reward_weight * action_sequence[self.history_size - 1 :] ** 2
        )
        expected_reward = particle_rewards.mean(axis=0).sum()
        return expected_reward.item()

    def sample_action_sequences(self):
        samples = np.random.multivariate_normal(
            mean=self.action_dist_mu,
            cov=self.action_dist_cov,
            size=self.num_action_samples,
        )
        return samples
