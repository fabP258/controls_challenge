from scipy import signal
from collections import deque


class DisturbanceObserver:

    def __init__(self, T_model: float, T_filter: float):

        # second order model, nonlinear gain assumed to be compensated by inverse nonlinearity
        self._T_model = T_model
        self._D_model = 1.0
        self._K_model = 1.0

        # second order filter
        self._T_filter = T_filter

        # inverse filtered plant discretization
        num_cont = [self._T_model**2, 2 * self._D_model * self._T_model, 1]
        den_cont = [
            self._K_model * self._T_filter**2,
            2 * self._K_model * self._T_filter,
            self._K_model,
        ]
        self._inverse_filtered_plant = signal.cont2discrete((num_cont, den_cont), dt=1)

        self._inverse_plant_u_buffer = deque([0.0, 0.0], maxlen=2)
        self._inverse_plant_y_buffer = deque([0.0, 0.0], maxlen=2)

        # action filter
        self._action_filter = signal.cont2discrete(
            ([1], [self._T_filter**2, 2 * self._T_filter, 1]), dt=1
        )

        # Note: Action is delayed by one step --> u_buffer size = 3
        self._action_filter_u_buffer = deque([0.0, 0.0, 0.0], maxlen=3)
        self._action_filter_y_buffer = deque([0.0, 0.0], maxlen=2)

    def update(self, measurement):
        """
        Updates the internal states and returns the corrective action
        """

        inv_plant_action = self.update_inverse_filtered_plant(measurement)

        filtered_action = self.update_action_filter(0.0)

        corrective_action = inv_plant_action - filtered_action

        return corrective_action

    def update_inverse_filtered_plant(self, measurement: float) -> float:
        """
        Update the state of the inverse filtered plant
        """
        num = self._inverse_filtered_plant[0][0]
        den = self._inverse_filtered_plant[1]

        y_k = (
            -den[1] * self._inverse_plant_y_buffer[0]
            - den[2] * self._inverse_plant_y_buffer[1]
            + num[0] * measurement
            + num[1] * self._inverse_plant_u_buffer[0]
            + num[2] * self._inverse_plant_u_buffer[1]
        )

        self._inverse_plant_y_buffer.pop()
        self._inverse_plant_y_buffer.appendleft(y_k)

        self._inverse_plant_u_buffer.pop()
        self._inverse_plant_u_buffer.appendleft(measurement)

        return y_k

    def update_action_filter(self, action: float) -> float:
        """
        Update the action filter state
        """

        num = self._action_filter[0][0]
        den = self._action_filter[1]

        y_k = (
            -den[1] * self._action_filter_y_buffer[0]
            - den[2] * self._action_filter_y_buffer[1]
            + num[0] * action
            + num[1] * self._action_filter_u_buffer[1]
            + num[2] * self._action_filter_u_buffer[2]
        )

        self._action_filter_y_buffer.pop()
        self._action_filter_y_buffer.appendleft(y_k)

        self._action_filter_u_buffer.pop()
        self._action_filter_u_buffer.appendleft(action)

        return y_k

    def store_action(self, action: float):
        """
        Applied action needs to be stored for action filter update in next cycle
        """
        self._action_filter_u_buffer.popleft()
        self._action_filter_u_buffer.appendleft(action)
