from copy import deepcopy

from numpy import ndarray
from pympc.utils import runge_kutta_4


class Model:
  def __init__(
      self,
      dynamics: callable,
      time_step: float,
      initial_state: ndarray,
      initial_actuation: ndarray,
      record: bool = False
      ):
    """
    :param dynamics: function that describes the dynamics of the system, must have the following
    signature: f(state, actuation, **kwargs)
    :param time_step: time step of the simulation
    :param initial_state: initial state of the system
    :param initial_actuation: initial actuation of the system
    :param kwargs: additional keyword arguments for dynamics
    :param record: whether to record the states and actuations
    """

    self.dynamics = dynamics
    self.time_step = time_step

    self.state = deepcopy( initial_state )
    self.actuation = deepcopy( initial_actuation )

    self.record = record
    if self.record:
      self.previous_states = [ deepcopy( self.state ) ]
      self.previous_actuations = [ deepcopy( self.actuation ) ]

  def step( self ):
    """
    integrates the system one time step. record the state and actuation if record is True
    """

    self.state = runge_kutta_4( self.dynamics, self.time_step, self.state, self.actuation )

    if self.record:
      self.previous_states.append( deepcopy( self.state ) )
      self.previous_actuations.append( deepcopy( self.actuation ) )
