from warnings import simplefilter

from numpy import dot, ndarray, r_, zeros
from numpy.linalg import norm

from pympc.models.catenary import Catenary
from pympc.models.dynamics.bluerov import BluerovXZPsi as Bluerov
from pympc.models.dynamics.dynamics import Dynamics

simplefilter( 'ignore', RuntimeWarning )


class ChainOf2( Dynamics ):

    _state_size = Bluerov().state_size * 2
    _actuation_size = Bluerov().actuation_size * 2

    _position = r_[ slice( 0, 3 ), slice( 6, 9 ) ]
    _orientation = r_[ slice( 3, 6 ), slice( 9, 12 ) ]
    _velocity = r_[ slice( 12, 15 ), slice( 18, 22 ) ]
    _body_rates = r_[ slice( 15, 18 ), slice( 22, 24 ) ]

    _linear_actuation = r_[
        Bluerov().linear_actuation, Bluerov().linear_actuation + Bluerov().linear_actuation.shape[ 0 ] +
                                    Bluerov().angular_actuation.shape[ 0 ] ]
    _angular_actuation = r_[
        Bluerov().angular_actuation, Bluerov().angular_actuation + Bluerov().linear_actuation.shape[ 0 ] +
                                     Bluerov().angular_actuation.shape[ 0 ] ]

    _br_state_size = Bluerov().state_size
    _br_actuation_size = Bluerov().actuation_size

    _br_0_position = _position[ :3 ]
    _br_0_orientation = _orientation[ :3 ]
    _br_0_velocity = _velocity[ :3 ]
    _br_0_body_rates = _body_rates[ :3 ]

    _br_0_pose = r_[ _br_0_position, _br_0_orientation ]
    _br_0_state = r_[ _br_0_position, _br_0_orientation, _br_0_velocity, _br_0_body_rates ]

    _br_0_linear_actuation = Bluerov().linear_actuation
    _br_0_angular_actuation = Bluerov().angular_actuation
    _br_0_actuation = r_[ _br_0_linear_actuation, _br_0_angular_actuation ]

    _br_0_perturbation = r_[ slice( 0, _state_size // (2 * 2) ) ]

    _br_1_position = _position[ 3: ]
    _br_1_orientation = _orientation[ 3: ]
    _br_1_velocity = _velocity[ 3: ]
    _br_1_body_rates = _body_rates[ 3: ]

    _br_1_pose = r_[ _br_1_position, _br_1_orientation ]
    _br_1_state = r_[ _br_1_position, _br_1_orientation, _br_1_velocity, _br_1_body_rates ]

    _br_1_linear_actuation = Bluerov().linear_actuation + Bluerov().actuation_size
    _br_1_angular_actuation = Bluerov().angular_actuation + Bluerov().actuation_size
    _br_1_actuation = r_[ _br_1_linear_actuation, _br_1_angular_actuation ]

    _br_1_perturbation = r_[ slice( _state_size // (2 * 2), _state_size // 2 ) ]

    def __init__(
        self,
        water_surface_depth: float = 0.,
        water_current: ndarray = None,
        cables_length: float = 3.,
        cables_linear_mass: float = 0.,
        get_cable_parameter_method = 'runtime',
        reference_frame: str = 'NED'
        ):

        # instanciate two bluerovs to be able to modify their parameters.
        # if both were identical we could just have one instance
        self.br_0 = Bluerov( water_surface_depth, water_current, reference_frame )
        self.c_01 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method, reference_frame )
        self.br_1 = Bluerov( water_surface_depth, water_current, reference_frame )

        self.last_perturbation_0 = zeros( (self.br_0.state_size // 2,) )
        self.last_perturbation_1 = zeros( (self.br_1.state_size // 2,) )

    def __call__( self, state: ndarray, actuation: ndarray, perturbation: ndarray ) -> ndarray:
        """
        evaluates the dynamics of the chain
        :param state: current state of the system
        :param actuation: current actuation of the system
        :return: state derivative of the system
        """
        state_derivative = zeros( state.shape )

        perturbation_01_0, perturbation_01_1 = self.c_01.get_perturbations(
            state[ self.br_0_position ], state[ self.br_1_position ]
            )

        # if the cable is taunt the perturbation is None
        # here we should consider any pair with a taunt cable as a single body
        if perturbation_01_0 is not None:
            self.last_perturbation_0[ :3 ] = perturbation_01_0
            self.last_perturbation_1[ :3 ] = perturbation_01_1
        else:
            perturbation_01_0, perturbation_01_1 = self.get_taunt_cable_perturbations( state, actuation )

        perturbation_01_0.resize( (self.br_state_size // 2,), refcheck = False )
        perturbation_01_1.resize( (self.br_state_size // 2,), refcheck = False )

        # cable perturbation is in world frame, should be applied robot frame instead
        br_0_transformation_matrix = self.br_0.build_transformation_matrix( *state[ self.br_0_orientation ] )
        br_1_transformation_matrix = self.br_1.build_transformation_matrix( *state[ self.br_1_orientation ] )
        perturbation_01_0 = br_0_transformation_matrix.T @ perturbation_01_0
        perturbation_01_1 = br_1_transformation_matrix.T @ perturbation_01_1

        state_derivative[ self.br_0_state ] = self.br_0(
            state[ self.br_0_state ],
            actuation[ self.br_0_actuation ],
            perturbation[ self._br_0_perturbation ] + perturbation_01_0
            )
        state_derivative[ self.br_1_state ] = self.br_1(
            state[ self.br_1_state ],
            actuation[ self.br_1_actuation ],
            perturbation[ self._br_1_perturbation ] + perturbation_01_1
            )

        return state_derivative

    def compute_error( self, actual: ndarray, target: ndarray ) -> ndarray:
        error = zeros( actual.shape )
        error[ :, :, self.br_0_pose ] = self.br_0.compute_error(
            actual[ :, :, self.br_0_pose ], target[ :, :, self.br_0_pose ]
            )
        error[ :, :, self.br_1_pose ] = self.br_1.compute_error(
            actual[ :, :, self.br_1_pose ], target[ :, :, self.br_1_pose ]
            )
        return error

    def get_taunt_cable_perturbations( self, state: ndarray, actuation: ndarray ) -> tuple:

        direction = state[ self.br_1_position ] - state[ self.br_0_position ]
        direction /= norm( direction )

        # we dont consider system-external perturbation because they are applied in __call__
        null = zeros( (self.br_state_size // 2,) )

        br_0_transformation_matrix = self.br_0.build_transformation_matrix( *state[ self.br_0_orientation ] )
        br_1_transformation_matrix = self.br_1.build_transformation_matrix( *state[ self.br_1_orientation ] )

        # in robot frame
        br_0_acceleration = self.br_0( state[ self.br_0_state ], actuation[ self.br_0_actuation ], null )[ 6: ]
        br_0_forces = self.br_0.inertial_matrix[ :3, :3 ] @ br_0_acceleration[ :3 ]

        br_1_acceleration = self.br_1( state[ self.br_1_state ], actuation[ self.br_1_actuation ], null )[ 6: ]
        br_1_forces = (self.br_1.inertial_matrix[ :3, :3 ] @ br_1_acceleration[ :3 ])

        all_forces = dot( br_0_transformation_matrix[ :3, :3 ] @ br_0_forces, -direction )
        all_forces += dot( br_1_transformation_matrix[ :3, :3 ] @ br_1_forces, direction )

        perturbation = direction * all_forces

        # in world frame
        return perturbation, -perturbation

    @property
    def br_state_size( self ):
        return self._br_state_size

    @property
    def br_actuation_size( self ):
        return self._br_actuation_size

    @property
    def br_0_position( self ):
        return self._br_0_position

    @property
    def br_0_orientation( self ):
        return self._br_0_orientation

    @property
    def br_0_velocity( self ):
        return self._br_0_velocity

    @property
    def br_0_body_rates( self ):
        return self._br_0_body_rates

    @property
    def br_0_state( self ):
        return self._br_0_state

    @property
    def br_0_pose( self ):
        return self._br_0_pose

    @property
    def br_0_actuation( self ):
        return self._br_0_actuation

    @property
    def br_0_perturbation( self ):
        return self._br_0_perturbation

    @property
    def br_0_linear_actuation( self ):
        return self._br_0_linear_actuation

    @property
    def br_0_angular_actuation( self ):
        return self._br_0_angular_actuation

    @property
    def br_1_position( self ):
        return self._br_1_position

    @property
    def br_1_orientation( self ):
        return self._br_1_orientation

    @property
    def br_1_velocity( self ):
        return self._br_1_velocity

    @property
    def br_1_body_rates( self ):
        return self._br_1_body_rates

    @property
    def br_1_state( self ):
        return self._br_1_state

    @property
    def br_1_pose( self ):
        return self._br_1_pose

    @property
    def br_1_actuation( self ):
        return self._br_1_actuation

    @property
    def br_1_perturbation( self ):
        return self._br_1_perturbation

    @property
    def br_1_linear_actuation( self ):
        return self._br_1_linear_actuation

    @property
    def br_1_angular_actuation( self ):
        return self._br_1_angular_actuation


if __name__ == '__main__':
    from numpy import set_printoptions
    from numpy.random import random

    set_printoptions( precision = 2, linewidth = 10000, suppress = True )

    ch2 = ChainOf2()

    print( f"{ch2.state_size=}" )
    print( f"{ch2.actuation_size=}" )
    print( f"{ch2.position=}" )
    print( f"{ch2.orientation=}" )
    print( f"{ch2.velocity=}" )
    print( f"{ch2.body_rates=}" )
    print( f"{ch2.linear_actuation=}" )
    print( f"{ch2.angular_actuation=}" )

    print( f"{ch2.br_state_size=}" )
    print( f"{ch2.br_actuation_size=}" )
    print( f"{ch2.br_0_position=}" )
    print( f"{ch2.br_0_orientation=}" )
    print( f"{ch2.br_0_velocity=}" )
    print( f"{ch2.br_0_body_rates=}" )
    print( f"{ch2.br_0_state=}" )
    print( f"{ch2.br_0_actuation=}" )
    print( f"{ch2.br_0_perturbation=}" )
    print( f"{ch2.br_0_linear_actuation=}" )
    print( f"{ch2.br_0_angular_actuation=}" )
    print( f"{ch2.br_1_position=}" )
    print( f"{ch2.br_1_orientation=}" )
    print( f"{ch2.br_1_velocity=}" )
    print( f"{ch2.br_1_body_rates=}" )
    print( f"{ch2.br_1_state=}" )
    print( f"{ch2.br_1_actuation=}" )
    print( f"{ch2.br_1_perturbation=}" )
    print( f"{ch2.br_1_linear_actuation=}" )
    print( f"{ch2.br_1_angular_actuation=}" )

    s = random( (ch2.state_size,) )
    a = random( (ch2.actuation_size,) )
    p = random( (ch2.state_size // 2,) )
    ds = ch2( s, a, p )
    s[ch2.br_0_position] += 0.
    s[ch2.br_1_position] += 5.
    ds = ch2( s, a, p )

    t = random( (10, 1, ch2.state_size // 2) )
    a = random( (10, 1, ch2.state_size // 2) )
    e = ch2.compute_error( a, t )
    print( f"{e=}" )
