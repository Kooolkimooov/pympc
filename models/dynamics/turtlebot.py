from numpy import cos, ndarray, pi, sin, zeros

from pympc.models.dynamics.dynamics import Dynamics


class Turtlebot( Dynamics ):
  """
  implementation of the dynamics of the Turtlebot model
  """
  _state_size = 6
  _actuation_size = 2

  _position = slice( 0, 2 )
  _orientation = 2
  _velocity = slice( 3, 5 )
  _body_rates = 5

  _linear_actuation = 0
  _angular_actuation = 1

  def __call__( self, state: ndarray, actuation: ndarray, perturbation: ndarray ) -> ndarray:
    xdot = zeros( (6,) )
    xdot[ 0 ] = cos( state[ 2 ] ) * actuation[ 0 ]
    xdot[ 1 ] = sin( state[ 2 ] ) * actuation[ 0 ]
    xdot[ 2 ] = actuation[ 1 ]

    return xdot

  def compute_error( self, actual: ndarray, target: ndarray ) -> ndarray:
    error = zeros( actual.shape )
    error[ :, :, :2 ] = actual[ :, :, :2 ] - target[ :, :, :2 ]
    error[ :, :, 2 ] = (actual[ :, :, 2 ] - target[ :, :, 2 ]) % pi

    return error
