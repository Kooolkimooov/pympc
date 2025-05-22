from numpy import dot, ndarray, r_, zeros
from numpy.linalg import norm

from pympc.controllers.mpc import MPC
from pympc.models.catenary import Catenary
from pympc.models.dynamics.bluerov import BluerovXYZPsi as Bluerov, USV
from pympc.models.dynamics.dynamics import Dynamics
from pympc.models.seafloor import Seafloor


class ChainOf4WithUSV( Dynamics ):
    _state_size = Bluerov().state_size * 4
    _actuation_size = Bluerov().actuation_size * 3 + USV().actuation_size

    _position = r_[ slice( 0, 3 ), slice( 6, 9 ), slice( 12, 15 ), slice( 18, 21 ) ]
    _orientation = r_[ slice( 3, 6 ), slice( 9, 12 ), slice( 15, 18 ), slice( 21, 24 ) ]
    _velocity = r_[ slice( 24, 27 ), slice( 30, 33 ), slice( 36, 39 ), slice( 42, 45 ) ]
    _body_rates = r_[ slice( 27, 30 ), slice( 33, 36 ), slice( 39, 42 ), slice( 45, 48 ) ]

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

    _br_0_perturbation = r_[ slice( 0 * _state_size // (2 * 4), 1 * _state_size // (2 * 4) ) ]

    _br_1_position = _position[ 3:6 ]
    _br_1_orientation = _orientation[ 3:6 ]
    _br_1_velocity = _velocity[ 3:6 ]
    _br_1_body_rates = _body_rates[ 3:6 ]

    _br_1_pose = r_[ _br_1_position, _br_1_orientation ]
    _br_1_state = r_[ _br_1_position, _br_1_orientation, _br_1_velocity, _br_1_body_rates ]

    _br_1_linear_actuation = Bluerov().linear_actuation + Bluerov().actuation_size
    _br_1_angular_actuation = Bluerov().angular_actuation + Bluerov().actuation_size
    _br_1_actuation = r_[ _br_1_linear_actuation, _br_1_angular_actuation ]

    _br_1_perturbation = r_[ slice( 1 * _state_size // (2 * 4), 2 * _state_size // (2 * 4) ) ]

    _br_2_position = _position[ 6:9 ]
    _br_2_orientation = _orientation[ 6:9 ]
    _br_2_velocity = _velocity[ 6:9 ]
    _br_2_body_rates = _body_rates[ 6:9 ]

    _br_2_pose = r_[ _br_2_position, _br_2_orientation ]
    _br_2_state = r_[ _br_2_position, _br_2_orientation, _br_2_velocity, _br_2_body_rates ]

    _br_2_linear_actuation = Bluerov().linear_actuation + Bluerov().actuation_size * 2
    _br_2_angular_actuation = Bluerov().angular_actuation + Bluerov().actuation_size * 2
    _br_2_actuation = r_[ _br_2_linear_actuation, _br_2_angular_actuation ]

    _br_2_perturbation = r_[ slice( 2 * _state_size // (2 * 4), 3 * _state_size // (2 * 4) ) ]

    _br_3_position = _position[ 9:12 ]
    _br_3_orientation = _orientation[ 9:12 ]
    _br_3_velocity = _velocity[ 9:12 ]
    _br_3_body_rates = _body_rates[ 9:12 ]

    _br_3_pose = r_[ _br_3_position, _br_3_orientation ]
    _br_3_state = r_[ _br_3_position, _br_3_orientation, _br_3_velocity, _br_3_body_rates ]

    _br_3_linear_actuation = USV().linear_actuation + Bluerov().actuation_size * 3
    _br_3_angular_actuation = USV().angular_actuation + Bluerov().actuation_size * 3
    _br_3_actuation = r_[ _br_3_linear_actuation, _br_3_angular_actuation ]

    _br_3_perturbation = r_[ slice( 3 * _state_size // (2 * 4), 4 * _state_size // (2 * 4) ) ]

    def __init__(
            self,
            water_surface_depth: float,
            water_current: ndarray,
            seafloor: Seafloor,
            cables_length: float,
            cables_linear_mass: float,
            get_cable_parameter_method,
            reference_frame
    ):
        self.br_0 = Bluerov( water_surface_depth, water_current, reference_frame )
        self.c_01 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method, reference_frame )
        self.br_1 = Bluerov( water_surface_depth, water_current, reference_frame )
        self.c_12 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method, reference_frame )
        self.br_2 = Bluerov( water_surface_depth, water_current, reference_frame )
        self.c_23 = Catenary( cables_length, cables_linear_mass, get_cable_parameter_method, reference_frame )
        self.br_3 = USV( water_surface_depth, reference_frame )

        self.sf = seafloor

    def __call__( self, state: ndarray, actuation: ndarray, perturbation: ndarray ) -> ndarray:
        """
        evaluates the dynamics of each robot of the chain
        :param state: current state of the system
        :param actuation: current actuation of the system
        :return: state derivative of the system
        """
        state_derivative = zeros( state.shape )

        perturbation_01_0, perturbation_01_1 = self.c_01.get_perturbations(
                state[ self.br_0_position ], state[ self.br_1_position ]
        )
        perturbation_12_1, perturbation_12_2 = self.c_12.get_perturbations(
                state[ self.br_1_position ], state[ self.br_2_position ]
        )
        perturbation_23_2, perturbation_23_3 = self.c_23.get_perturbations(
                state[ self.br_2_position ], state[ self.br_3_position ]
        )

        # if the cable is taunt the perturbation is None
        # here we should consider any pair with a taunt cable as a single body
        if perturbation_01_0 is None:
            perturbation_01_0, perturbation_01_1 = self.get_taunt_cable_perturbations(
                    self.br_0,
                    self.br_1,
                    state[ self.br_0_state ],
                    state[ self.br_1_state ],
                    actuation[ self.br_0_actuation ],
                    actuation[ self.br_1_actuation ]
            )

        if perturbation_12_1 is None:
            perturbation_12_1, perturbation_12_2 = self.get_taunt_cable_perturbations(
                    self.br_1,
                    self.br_2,
                    state[ self.br_1_state ],
                    state[ self.br_2_state ],
                    actuation[ self.br_1_actuation ],
                    actuation[ self.br_2_actuation ]
            )

        if perturbation_23_2 is None:
            perturbation_23_2, perturbation_23_3 = self.get_taunt_cable_perturbations(
                    self.br_2,
                    self.br_3,
                    state[ self.br_2_state ],
                    state[ self.br_3_state ],
                    actuation[ self.br_2_actuation ],
                    actuation[ self.br_3_actuation ]
            )

        perturbation_01_0.resize( (self.br_state_size // 2,), refcheck=False )
        perturbation_01_1.resize( (self.br_state_size // 2,), refcheck=False )
        perturbation_12_1.resize( (self.br_state_size // 2,), refcheck=False )
        perturbation_12_2.resize( (self.br_state_size // 2,), refcheck=False )
        perturbation_23_2.resize( (self.br_state_size // 2,), refcheck=False )
        perturbation_23_3.resize( (self.br_state_size // 2,), refcheck=False )

        # perturbation is in world frame, should be applied robot frame instead
        br_0_transformation_matrix = self.br_0.build_transformation_matrix( *state[ self.br_0_orientation ] )
        br_1_transformation_matrix = self.br_1.build_transformation_matrix( *state[ self.br_1_orientation ] )
        br_2_transformation_matrix = self.br_2.build_transformation_matrix( *state[ self.br_2_orientation ] )
        br_3_transformation_matrix = self.br_3.build_transformation_matrix( *state[ self.br_3_orientation ] )

        perturbation_01_0 = br_0_transformation_matrix.T @ perturbation_01_0
        perturbation_01_1 = br_1_transformation_matrix.T @ perturbation_01_1
        perturbation_12_1 = br_1_transformation_matrix.T @ perturbation_12_1
        perturbation_12_2 = br_2_transformation_matrix.T @ perturbation_12_2
        perturbation_23_2 = br_2_transformation_matrix.T @ perturbation_23_2
        perturbation_23_3 = br_3_transformation_matrix.T @ perturbation_23_3

        state_derivative[ self.br_0_state ] = self.br_0(
                state[ self.br_0_state ],
                actuation[ self.br_0_actuation ],
                perturbation[ self.br_0_perturbation ] + perturbation_01_0
        )
        state_derivative[ self.br_1_state ] = self.br_1(
                state[ self.br_1_state ],
                actuation[ self.br_1_actuation ],
                perturbation[ self.br_1_perturbation ] + perturbation_01_1 + perturbation_12_1
        )
        state_derivative[ self.br_2_state ] = self.br_2(
                state[ self.br_2_state ],
                actuation[ self.br_2_actuation ],
                perturbation[ self.br_2_perturbation ] + perturbation_12_2 + perturbation_23_2
        )
        state_derivative[ self.br_3_state ] = self.br_3(
                state[ self.br_3_state ],
                actuation[ self.br_3_actuation ],
                perturbation[ self.br_3_perturbation ] + perturbation_23_3
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
        error[ :, :, self.br_2_pose ] = self.br_2.compute_error(
                actual[ :, :, self.br_2_pose ], target[ :, :, self.br_2_pose ]
        )
        error[ :, :, self.br_3_pose ] = self.br_3.compute_error(
                actual[ :, :, self.br_3_pose ], target[ :, :, self.br_3_pose ]
        )
        return error

    def get_taunt_cable_perturbations(
            self,
            br_0: Bluerov,
            br_1: Bluerov,
            br_0_state: ndarray,
            br_1_state: ndarray,
            br_0_actuation: ndarray,
            br_1_actuation: ndarray
    ) -> tuple:
        # from br_0 to br_1
        direction = br_1_state[ br_1.position ] - br_0_state[ br_0.position ]
        direction /= norm( direction )

        null = zeros( (self.br_state_size // 2,) )

        br_0_transformation_matrix = br_0.build_transformation_matrix( *br_0_state[ br_0.orientation ] )
        br_1_transformation_matrix = br_1.build_transformation_matrix( *br_1_state[ br_1.orientation ] )

        # in robot frame
        br_0_acceleration = br_0( br_0_state, br_0_actuation, null )[ 6: ]
        br_0_forces = br_0.inertial_matrix[ :3, :3 ] @ br_0_acceleration[ :3 ]

        br_1_acceleration = br_1( br_1_state, br_1_actuation, null )[ 6: ]
        br_1_forces = br_1.inertial_matrix[ :3, :3 ] @ br_1_acceleration[ :3 ]

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
    def br_0_pose( self ):
        return self._br_0_pose

    @property
    def br_0_state( self ):
        return self._br_0_state

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
    def br_1_pose( self ):
        return self._br_1_pose

    @property
    def br_1_state( self ):
        return self._br_1_state

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

    @property
    def br_2_position( self ):
        return self._br_2_position

    @property
    def br_2_orientation( self ):
        return self._br_2_orientation

    @property
    def br_2_velocity( self ):
        return self._br_2_velocity

    @property
    def br_2_body_rates( self ):
        return self._br_2_body_rates

    @property
    def br_2_pose( self ):
        return self._br_2_pose

    @property
    def br_2_state( self ):
        return self._br_2_state

    @property
    def br_2_linear_actuation( self ):
        return self._br_2_linear_actuation

    @property
    def br_2_angular_actuation( self ):
        return self._br_2_angular_actuation

    @property
    def br_2_actuation( self ):
        return self._br_2_actuation

    @property
    def br_2_perturbation( self ):
        return self._br_2_perturbation

    @property
    def br_3_position( self ):
        return self._br_3_position

    @property
    def br_3_orientation( self ):
        return self._br_3_orientation

    @property
    def br_3_velocity( self ):
        return self._br_3_velocity

    @property
    def br_3_body_rates( self ):
        return self._br_3_body_rates

    @property
    def br_3_pose( self ):
        return self._br_3_pose

    @property
    def br_3_state( self ):
        return self._br_3_state

    @property
    def br_3_linear_actuation( self ):
        return self._br_3_linear_actuation

    @property
    def br_3_angular_actuation( self ):
        return self._br_3_angular_actuation

    @property
    def br_3_actuation( self ):
        return self._br_3_actuation

    @property
    def br_3_perturbation( self ):
        return self._br_3_perturbation


def chain_of_4_constraints( self: MPC, candidate: ndarray ) -> ndarray:
    chain: ChainOf4WithUSV = self.model.dynamics

    actuation, _ = self.get_actuation( candidate )

    prediction = self.predict( actuation )
    prediction = prediction[ :, 0 ]

    # 3 constraints on cables (distance of lowest point to seafloor)
    # 4 constraints on robots (distance of lowest point to seafloor)
    # 6 on inter robot_distance (3 horizontal, 2 3d)
    n_constraints = 3 + 6
    constraints = zeros( (self.horizon, n_constraints) )

    # eliminate this loop
    # ???
    # profit
    for i, state in enumerate( prediction ):
        c01 = chain.c_01.discretize( state[ chain.br_0_position ], state[ chain.br_1_position ], 10 )
        c12 = chain.c_12.discretize( state[ chain.br_1_position ], state[ chain.br_2_position ], 10 )
        c23 = chain.c_23.discretize( state[ chain.br_2_position ], state[ chain.br_3_position ], 10 )

        # cables distance from seafloor [0, 3[
        constraints[ i, 0 ] = min( [ chain.sf.get_distance_to_seafloor( p ) for p in c01 ] )
        constraints[ i, 1 ] = min( [ chain.sf.get_distance_to_seafloor( p ) for p in c12 ] )
        constraints[ i, 2 ] = min( [ chain.sf.get_distance_to_seafloor( p ) for p in c23 ] )

    # horizontal distance between consecutive robots [7, 10[
    constraints[ :, 3 ] = norm(
            prediction[ :, chain.br_1_position[ :2 ] ] - prediction[ :, chain.br_0_position[ :2 ] ], axis=1
    )
    constraints[ :, 4 ] = norm(
            prediction[ :, chain.br_2_position[ :2 ] ] - prediction[ :, chain.br_1_position[ :2 ] ], axis=1
    )
    constraints[ :, 5 ] = norm(
            prediction[ :, chain.br_3_position[ :2 ] ] - prediction[ :, chain.br_2_position[ :2 ] ], axis=1
    )

    # distance between consecutive robots [10, 13[
    constraints[ :, 6 ] = norm(
            prediction[ :, chain.br_1_position ] - prediction[ :, chain.br_0_position ], axis=1
    )
    constraints[ :, 7 ] = norm(
            prediction[ :, chain.br_2_position ] - prediction[ :, chain.br_1_position ], axis=1
    )
    constraints[ :, 8 ] = norm(
            prediction[ :, chain.br_3_position ] - prediction[ :, chain.br_2_position ], axis=1
    )

    return constraints.flatten()


def chain_of_4_objective( self: MPC, prediction: ndarray, actuation: ndarray ) -> float:
    chain: ChainOf4WithUSV = self.model.dynamics
    desired_distance = chain.c_01.length / 2

    objective = 0.

    # objective += pow( norm( prediction[ :, 0, chain.br_0_velocity ], axis = 1 ).sum(), 2 )
    objective += pow( norm( prediction[ :, 0, chain.br_1_velocity ], axis=1 ).sum(), 2 )
    objective += pow( norm( prediction[ :, 0, chain.br_2_velocity ], axis=1 ).sum(), 2 )
    objective += pow( norm( prediction[ :, 0, chain.br_3_velocity ], axis=1 ).sum(), 2 )

    objective += abs(
            norm(
                    prediction[ :, 0, chain.br_0_position ] - prediction[ :, 0, chain.br_1_position ], axis=1
            ) - desired_distance
    ).sum()
    objective += abs(
            norm(
                    prediction[ :, 0, chain.br_1_position ] - prediction[ :, 0, chain.br_2_position ], axis=1
            ) - desired_distance
    ).sum()
    objective += abs(
            norm(
                    prediction[ :, 0, chain.br_2_position ] - prediction[ :, 0, chain.br_3_position ], axis=1
            ) - desired_distance
    ).sum()

    objective /= self.horizon
    return objective


if __name__ == '__main__':
    from numpy import set_printoptions
    from numpy.random import random

    set_printoptions( precision=2, linewidth=10000, suppress=True )

    ch4 = ChainOf4WithUSV( 0.0, zeros( (3,) ), None, 0.0, 0.0, "runtime", reference_frame="NED" )

    print( f"{ch4.state_size=}" )
    print( f"{ch4.actuation_size=}" )
    print( f"{ch4.position=}" )
    print( f"{ch4.orientation=}" )
    print( f"{ch4.velocity=}" )
    print( f"{ch4.body_rates=}" )
    print( f"{ch4.linear_actuation=}" )
    print( f"{ch4.angular_actuation=}" )

    print( f"{ch4.br_state_size=}" )
    print( f"{ch4.br_actuation_size=}" )
    print( f"{ch4.br_0_position=}" )
    print( f"{ch4.br_0_orientation=}" )
    print( f"{ch4.br_0_velocity=}" )
    print( f"{ch4.br_0_body_rates=}" )
    print( f"{ch4.br_0_state=}" )
    print( f"{ch4.br_0_actuation=}" )
    print( f"{ch4.br_0_perturbation=}" )
    print( f"{ch4.br_0_linear_actuation=}" )
    print( f"{ch4.br_0_angular_actuation=}" )
    print( f"{ch4.br_1_position=}" )
    print( f"{ch4.br_1_orientation=}" )
    print( f"{ch4.br_1_velocity=}" )
    print( f"{ch4.br_1_body_rates=}" )
    print( f"{ch4.br_1_state=}" )
    print( f"{ch4.br_1_actuation=}" )
    print( f"{ch4.br_1_perturbation=}" )
    print( f"{ch4.br_1_linear_actuation=}" )
    print( f"{ch4.br_1_angular_actuation=}" )
    print( f"{ch4.br_2_position=}" )
    print( f"{ch4.br_2_orientation=}" )
    print( f"{ch4.br_2_velocity=}" )
    print( f"{ch4.br_2_body_rates=}" )
    print( f"{ch4.br_2_state=}" )
    print( f"{ch4.br_2_actuation=}" )
    print( f"{ch4.br_2_perturbation=}" )
    print( f"{ch4.br_2_linear_actuation=}" )
    print( f"{ch4.br_2_angular_actuation=}" )
    print( f"{ch4.br_3_position=}" )
    print( f"{ch4.br_3_orientation=}" )
    print( f"{ch4.br_3_velocity=}" )
    print( f"{ch4.br_3_body_rates=}" )
    print( f"{ch4.br_3_state=}" )
    print( f"{ch4.br_3_actuation=}" )
    print( f"{ch4.br_3_perturbation=}" )
    print( f"{ch4.br_3_linear_actuation=}" )
    print( f"{ch4.br_3_angular_actuation=}" )

    s = random( (ch4.state_size,) )
    a = random( (ch4.actuation_size,) )
    p = random( (ch4.state_size // 2,) )
    ds = ch4( s, a, p )

    t = random( (10, 1, ch4.state_size // 2) )
    a = random( (10, 1, ch4.state_size // 2) )
    e = ch4.compute_error( a, t )
    print( f"{e=}" )
