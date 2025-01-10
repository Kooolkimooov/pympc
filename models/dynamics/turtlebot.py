from numpy import cos, ndarray, sin, zeros


class Turtlebot:
  """
  Turtlebot model
  """

  state_size = 3
  pose_size = 2
  actuation_size = 2
  linear_actuation_size = 1

  def __init__( self ):
    pass

  def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
    """
    evaluates the dynamics of the Turtlebot model
    :param state: current state of the system
    :param actuation: current actuation of the system
    :return: state derivative of the system
    """

    xdot = zeros( (3,) )
    xdot[ 0 ] = cos( state[ 2 ] ) * actuation[ 0 ]
    xdot[ 1 ] = sin( state[ 2 ] ) * actuation[ 0 ]
    xdot[ 2 ] = actuation[ 1 ]

    return xdot
