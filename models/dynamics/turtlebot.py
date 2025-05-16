from numpy import cos, ndarray, pi, r_, sin, zeros

from pympc.models.dynamics.dynamics import Dynamics


class Turtlebot( Dynamics ):
    """
    Implements the dynamics of the Turtlebot model
    """

    _state_size = 6
    _actuation_size = 2

    _position = r_[ slice( 0, 2 ) ]
    _orientation = r_[ 2 ]
    _velocity = r_[ slice( 3, 5 ) ]
    _body_rates = r_[ 5 ]

    _linear_actuation = r_[ 0 ]
    _angular_actuation = r_[ 1 ]

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
    from numpy import set_printoptions
    from numpy.random import random

    set_printoptions( precision = 2, linewidth = 10000, suppress = True )

    tb = Turtlebot()

    print( f"{tb.state_size=}" )
    print( f"{tb.actuation_size=}" )
    print( f"{tb.position=}" )
    print( f"{tb.orientation=}" )
    print( f"{tb.velocity=}" )
    print( f"{tb.body_rates=}" )
    print( f"{tb.linear_actuation=}" )
    print( f"{tb.angular_actuation=}" )

    s = random( (tb.state_size,) )
    a = random( (tb.actuation_size,) )
    p = random( (tb.state_size // 2,) )
    ds = tb( s, a, p )

    t = random( (10, 1, tb.state_size // 2) )
    a = random( (10, 1, tb.state_size // 2) )
    e = tb.compute_error( a, t )
