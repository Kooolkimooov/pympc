from numpy import ndarray, zeros

from pympc.models.dynamics.dynamics import Dynamics
from pympc.utils import runge_kutta_4


class Model:
    """
    wrapper class for dynamics that stores the state, integrates the system and with ability to record the states and
    actuations

    Parameters
    ----------
    dynamics: Dynamics 
        function that describes the dynamics of the system, must have the following signature: f(state, actuation, **kwargs)
    time_step: float
        time step of the simulation
    initial_state: ndarray
        initial state of the system
    initial_actuation: ndarray
        initial actuation of the system
    initial_perturbation: ndarray
        initial perturbation of the system
    record: bool
        whether to record the states and actuations

    Methods
    -------
    **step**():
        integrates the system one time step. record the state and actuation if record is True

    Attributes
    ----------
    dynamics: Dynamics 
        function that describes the dynamics of the system
    time_step: float
        time step of the simulation
    state: ndarray
        current state of the system
    actuation: ndarray
        current actuation of the system
    perturbation: ndarray
        current perturbation of the system
    record: bool
        whether to record the states and actuations
    previous_states: list[ndarray] 
        list of previous states
    previous_actuations: list[ndarray]
        list of previous actuations
    previous_perturbations: list[ndarray]
        list of previous perturbations
    """

    def __init__(
            self,
            dynamics: Dynamics,
            time_step: float,
            initial_state: ndarray = None,
            initial_actuation: ndarray = None,
            initial_perturbation: ndarray = None,
            record: bool = False
    ):
        self.dynamics = dynamics
        self.time_step = time_step

        self.state = initial_state.copy() if not initial_state is None else zeros( (dynamics.state_size,) )
        self.actuation = initial_actuation.copy() if not initial_actuation is None else zeros(
                (dynamics.actuation_size,)
        )
        self.perturbation = initial_perturbation.copy() if not initial_perturbation is None else zeros(
                (dynamics.state_size // 2,)
        )

        self.record = record
        if self.record:
            self.previous_states = [ self.state.copy() ]
            self.previous_actuations = [ self.actuation.copy() ]
            self.previous_perturbations = [ self.perturbation.copy() ]

    def step( self ):
        """
        integrates the system one time step. record the state and actuation if record is True
        """

        self.state = runge_kutta_4(
                self.dynamics, self.time_step, self.state, actuation=self.actuation, perturbation=self.perturbation
        )

        if self.record:
            self.previous_states.append( self.state.copy() )
            self.previous_actuations.append( self.actuation.copy() )
            self.previous_perturbations.append( self.perturbation.copy() )
