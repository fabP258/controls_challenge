import numpy as np
import torch
from . import BaseController
import timeit
from collections import deque
import matplotlib.pyplot as plt


class Controller(BaseController):

    def __init__(self, models: list):
        self.models = models
        self.history_size = 1
        self.T = 10
        self.num_particles = 10
        self.num_action_samples = 50
        self.num_elites = 5
        self.action_reward_weight = 1.0
        self.num_cem_updates = 10

        # cross entropy distribution
        self.action_dist_mu = 0.0 * np.ones(((self.T),))
        self.action_dist_cov = np.eye((self.T)) * 0.5

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

        # start = timeit.default_timer()

        lat_accel_setpoints = target_lataccel * torch.ones(
            (self.T + 1)
        )  # TODO: use future_plan
        if len(lat_accel_setpoints) != (self.T + 1):
            raise ValueError("Size mismatch of state setpoints")

        self.lat_accel_history.popleft()
        self.lat_accel_history.append(current_lataccel)

        # colors = ["r", "g", "b", "c", "m"]
        # fig, ax = plt.subplots(nrows=2)

        for j in range(self.num_cem_updates):
            action_sequences = self.sample_action_sequences()
            action_sequences = torch.tensor(action_sequences)
            # TODO: Attach action history

            state_trajectories = torch.zeros(
                (
                    self.num_particles * self.num_action_samples,
                    self.T + self.history_size,
                )
            )
            state_trajectories[:, : self.history_size] = torch.tensor(
                self.lat_accel_history
            )
            bootstrap_indices = np.random.randint(
                low=0,
                high=len(self.models),
                size=self.num_particles * self.num_action_samples,
            )
            for t in range(self.T):
                x = torch.zeros(
                    (
                        self.num_particles * self.num_action_samples,
                        2 * self.history_size,
                    ),
                    dtype=torch.float32,
                )
                # TODO: avoid this loop
                for i in range(self.num_action_samples):
                    x[
                        i * self.num_particles : (i + 1) * self.num_particles,
                        : self.history_size,
                    ] = action_sequences[i][t]
                x[:, self.history_size :] = state_trajectories[
                    :, t : t + self.history_size
                ]
                for model_idx in range(len(self.models)):
                    indices = bootstrap_indices == model_idx
                    x_batch = x[indices]
                    with torch.no_grad():
                        pred_mean, pred_log_var = self.models[model_idx](x_batch)
                        std = torch.exp(0.5 * pred_log_var)
                        eps = torch.randn_like(std)
                        sampled_states = pred_mean + eps * std
                    state_trajectories[indices, t + self.history_size] = (
                        sampled_states.squeeze()
                    )
            particle_rewards = -(
                (state_trajectories[:, self.history_size - 1 :] - lat_accel_setpoints)
                ** 2
            )
            expanded_action_rewards = np.repeat(
                -self.action_reward_weight
                * action_sequences[self.history_size - 1 :] ** 2,
                repeats=self.num_particles,
                axis=0,
            )
            particle_rewards[:, : -self.history_size] += expanded_action_rewards
            action_rewards = (
                particle_rewards.reshape(
                    self.num_action_samples, self.num_particles, -1
                )
                .mean(axis=1)
                .sum(axis=1)
            )

            # for plotting
            # mean_state_trajectories = (
            #    state_trajectories.reshape(
            #        self.num_action_samples, self.num_particles, -1
            #    )
            #    .mean(axis=1)
            #    .squeeze()
            # )
            # for traj in mean_state_trajectories:
            #    ax[0].plot(traj.detach().numpy(), color=colors[j % len(colors)])
            # for actions in action_sequences:
            #    ax[1].stairs(actions.detach().numpy(), color=colors[j % len(colors)])

            if j == (self.num_cem_updates - 1):
                best_action_idx = torch.argmax(action_rewards)
                best_action_sequence = action_sequences[best_action_idx]
                action = best_action_sequence[0]
                # ax[0].plot(
                #    mean_state_trajectories[best_action_idx].detach().numpy(), color="k"
                # )
                # ax[1].stairs(best_action_sequence.detach().numpy(), color="k")
                # ax[0].plot(lat_accel_setpoints.numpy(), linestyle="--", color="k")
                # fig.savefig("cem_mpc.png")
                # plt.close(fig)
                if self.history_size > 1:
                    self.action_history.popleft()
                    self.action_history.append(action)
                # end = timeit.default_timer()
                # print(f"PETS-MPC took: {end - start} s")
                # TODO: shift mean left and pad
                next_action_dist_mean = np.empty_like(self.action_dist_mu)
                next_action_dist_mean[:-1] = self.action_dist_mu[1:]
                next_action_dist_mean[-1] = 0
                self.action_dist_cov = np.eye((self.T)) * 0.5
                return action

            # update CEM distribution
            elite_indices = action_rewards.argsort()[-self.num_elites :]
            elite_actions = action_sequences[elite_indices]
            self.action_dist_mu = elite_actions.mean(axis=0)
            self.action_dist_cov = np.cov(elite_actions.T) + 1e-6 * np.eye(
                elite_actions.shape[1]
            )

    def sample_action_sequences(self):
        samples = np.random.multivariate_normal(
            mean=self.action_dist_mu,
            cov=self.action_dist_cov,
            size=self.num_action_samples,
        )
        return samples
