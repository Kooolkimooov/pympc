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


if __name__ == '__main__':
  turtle_bot = Turtlebot()
  print( f"{turtle_bot.state_size=}" )
  print( f"{turtle_bot.actuation_size=}" )
  print( f"{turtle_bot.position=}" )
  print( f"{turtle_bot.orientation=}" )
  print( f"{turtle_bot.velocity=}" )
  print( f"{turtle_bot.body_rates=}" )
  print( f"{turtle_bot.linear_actuation=}" )
  print( f"{turtle_bot.angular_actuation=}" )

  from numpy import ones

  state = zeros( (turtle_bot.state_size,) )
  actuation = ones( (turtle_bot.actuation_size,) )

  print( f"{turtle_bot(state, actuation)=}" )

  t1 = zeros( (1, 1, turtle_bot.state_size // 2) )
  t2 = 10 * ones( (1, 1, turtle_bot.state_size // 2) )

  print( turtle_bot.compute_error( t1, t2 ) )
