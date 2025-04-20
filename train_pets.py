from tqdm import tqdm
import numpy as np
from dynamics_dataset import RolloutData, DynamicsDataset
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX
from controllers import pid, pets_mpc
from controllers import probabilistic_forward_dynamics as pfd
from ensemble_trainer import EnsembleTrainer
import matplotlib.pyplot as plt

model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=True)
history_size = 1
dataset = DynamicsDataset(history_size=history_size)
num_rollouts = 10
num_pets_iterations = 1
num_train_epochs = 5
num_ensembles = 5


def analyze_lag(x: np.array, y: np.array):
    lags = np.arange(-len(x) + 1, len(x))
    cross_corr = np.correlate(x - np.mean(x), y - np.mean(y), mode="full") / (
        np.std(x) * np.std(y) * len(x)
    )

    # Plotting
    plt.plot(lags, cross_corr)
    plt.title("Cross-Correlation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation Coefficient")
    plt.grid(True)
    plt.show()

    max_corr_lag = lags[np.argmax(cross_corr)]
    print("Max correlation at lag:", max_corr_lag)


# Phase 1: Initialization of Dataset D with PID controller
for i in tqdm(range(num_rollouts), desc="Generating initial dataset with PID"):
    file_name = f"./data/{i:05d}.csv"
    controller = pid.Controller()
    sim = TinyPhysicsSimulator(model, file_name, controller=controller, debug=False)
    sim.rollout()
    after_control_start = np.arange(len(sim.action_history)) >= CONTROL_START_IDX
    rollout_data = RolloutData(
        lat_accel=np.array(sim.current_lataccel_history)[after_control_start],
        steer=np.array(sim.action_history)[after_control_start],
        v_ego=np.array([state.v_ego for state in sim.state_history])[
            after_control_start
        ],
    )
    dataset.add_rollout_data(rollout_data)
print(f"Initial dataset for forward dynamics has {len(dataset)} entries")

steer_data, lat_accel_data = dataset.get_raw_data()
fig, ax = plt.subplots()
ax.scatter(steer_data, lat_accel_data, alpha=0.5)
ax.set_xlabel("steer")
ax.set_ylabel("lat. accel")
fig.savefig("static_io.png")

ensemble_trainer = EnsembleTrainer(
    state_dim=1,
    action_dim=1,
    history_size=history_size,
    num_ensembles=num_ensembles,
    lr=1e-3,
    weight_decay=0,
    batch_size=128,
)
# Phase 2: On-policy rollouts with PETS
for i in range(num_pets_iterations):
    # Train the ensemble of probabilistic NN's
    ensemble_trainer.train(num_epochs=num_train_epochs, train_dataset=dataset)
    ensemble_trainer.save_models()
    # Run on-policy rollouts with model predictive controller
    for i in tqdm(range(num_rollouts), desc=f"Generating on-policy rollouts"):
        file_name = f"./data/{i:05d}.csv"
        controller = pets_mpc.Controller(models=ensemble_trainer.models)
        sim = TinyPhysicsSimulator(model, file_name, controller=controller, debug=False)
        sim.rollout()
        rollout_cots = sim.compute_cost()
        print(f"PETS Iter.: {i}, Rollout Idx.: {i}, Cost: {rollout_cots['total_cost']}")
        after_control_start = np.arange(len(sim.action_history)) >= CONTROL_START_IDX
        rollout_data = RolloutData(
            lat_accel=np.array(sim.current_lataccel_history)[after_control_start],
            steer=np.array(sim.action_history)[after_control_start],
            v_ego=np.array([state.v_ego for state in sim.state_history])[
                after_control_start
            ],
        )
        dataset.add_rollout_data(rollout_data)
