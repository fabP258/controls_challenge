from . import BaseController
from . import static_nonlinearity as sn
from . import disturbance_observer as do
from tinyphysics import CONTROL_START_IDX


class Controller(BaseController):
    """
    A disturbance observer based controller
    """

    def __init__(self, T_model: float = 1.2, T_filter: float = 2.1):
        # nonlinear mapping
        self._nonlinearity: sn.StaticNonlinearity = sn.ArctanNonlinearity(A=1.5, B=0.45)

        # disturbance observer
        self._disturbance_observer = do.DisturbanceObserver(
            T_model=T_model, T_filter=T_filter
        )

        # lag compensation
        self._ffw_preview_steps: int = 2

        # step counter for init
        self.step = 0

        # DEBUG
        self.linear_action_history = []

        # TODO: increase u buffer in DOB

    def update(self, target_lataccel, current_lataccel, state, future_plan):

        # feedforward control
        feedforward = target_lataccel

        # compensate for lag if future plan is available
        if self._ffw_preview_steps > 0:
            future_plan_index = self._ffw_preview_steps - 1
            if len(future_plan.lataccel) > future_plan_index:
                feedforward = future_plan.lataccel[future_plan_index]

        linear_action = feedforward

        # store debug signals
        self.linear_action_history.append(linear_action)

        # calculate disturbance observer
        corrective_action = 0.0
        if self.step > (CONTROL_START_IDX - 20):
            corrective_action = self._disturbance_observer.update(current_lataccel)

        linear_action -= corrective_action

        # set u buffer in DOB
        self._disturbance_observer.store_action(linear_action)

        # add roll compensation
        linear_action -= state.roll_lataccel

        self.step += 1

        return self._nonlinearity.calculate_torque_from_lataccel(linear_action)
