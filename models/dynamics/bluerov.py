from numpy import array, concatenate, cos, cross, diag, exp, eye, ndarray, pi, r_, sin, tan, zeros
from numpy.linalg import inv

from pympc.models.dynamics.dynamics import Dynamics
from pympc.utils import G, rho_eau


class Bluerov( Dynamics ):
    """
    implementation of the Bluerov model, based on the BlueROV model from Blue Robotics
    parameters of the model are based on the BlueROV2 Heavy configuration
    and are stored in the class as class variables

    Parameters
    ----------
    water_surface_depth: float
        depth of the water surface
    water_current: ndarray
        water current in the x, y, z directions
    reference_frame: str
        reference frame of the model, either 'NED' or 'ENU'

    Methods
    -------
    **\_\_call\_\_**( *ndarray*, *ndarray*, *ndarray* ) -> *ndarray*:
        evaluates the dynamics
    **compute_error**( *ndarray*, *ndarray* ) -> *ndarray*:
        computes the error between two trajectories according to the system definition
    
    Properties
    ----------
    **state_size**: *int*:
        size of the expected state vector
    **actuation_size**: *int*:
        size of the expected actuation vector
    **position**: *ndarray*:
        indices of the position inside the state vector
    **orientation**: *ndarray*:
        indices of the orientation inside the state vector
    **velocity**: *ndarray*:
        indices of the velocity inside the state vector
    **body_rates**: *ndarray*:
        indices of the body rates inside the state vector
    **linear_actuation**: *ndarray*:
        indices of the linear actuation inside the actuation vector
    **angular_actuation**: *ndarray*:
        indices of the angular actuation inside the actuation vector
    **six_dof_actuation_mask**: *ndarray*:
        relation between a full six degrees of freedom actuation and the actuation of the model
    """

    _state_size = 12
    _actuation_size = 6

    _position = r_[ slice( 0, 3 ) ]
    _orientation = r_[ slice( 3, 6 ) ]
    _velocity = r_[ slice( 6, 9 ) ]
    _body_rates = r_[ slice( 9, 12 ) ]

    _linear_actuation = r_[ slice( 0, 3 ) ]
    _angular_actuation = r_[ slice( 3, 6 ) ]

    _six_dof_actuation_mask = r_[ slice( 0, 6 ) ]

    REFERENCE_FRAME = [ 'NED', 'ENU' ]

    def __init__( self, water_surface_depth: float = 0., water_current: ndarray = None, reference_frame: str = 'NED' ):
        if reference_frame == 'NED':
            self.vertical_multiplier = -1.
        elif reference_frame == 'ENU':
            self.vertical_multiplier = 1.
        else:
            raise ValueError( f'reference_frame must be one of {self.REFERENCE_FRAME}' )

        self.mass = 11.5
        self.center_of_mass = self.vertical_multiplier * array( [ 0.0, 0.0, 0.0 ] )
        self.weight = -self.vertical_multiplier * array( [ 0., 0., self.mass * G ] )

        self.volume = 0.0134
        self.center_of_volume = self.vertical_multiplier * array( [ 0.0, 0.0, 0.01 ] )
        self.buoyancy = self.vertical_multiplier * array( [ 0., 0., rho_eau * G * self.volume ] )

        self.water_surface_depth = water_surface_depth

        # water speed should be on [3:6]
        water_current = zeros( (6,) ) if water_current is None else water_current
        if water_current.shape == (3,):
            water_current = concatenate( (water_current, array( [ 0., 0., 0. ] )) )

        assert water_current.shape == (6,), 'water current should be a (6,) ndarray at this point'

        self.water_current = water_current

        self.inertial_coefficients = [ .26, .23, .37, 0., 0., 0. ]
        self.hydrodynamic_coefficients = [ 13.7, 13.7, 33.0, 0.8, 0.8, 0.8 ]
        self.added_mass_coefficients = [ 6.36, 7.12, 18.68, .189, .135, .222 ]

        self.inertial_matrix = self.build_inertial_matrix(
                self.mass, self.center_of_mass, self.inertial_coefficients
        ) + diag( self.added_mass_coefficients )

        self.inverse_inertial_matrix = inv( self.inertial_matrix )

        self.hydrodynamic_matrix = diag( self.hydrodynamic_coefficients )

    def __call__( self, state: ndarray, actuation: ndarray, perturbation: ndarray ) -> ndarray:
        transform_matrix = self.get_body_to_world_transform( state )

        six_dof_actuation = zeros( (6,) )
        six_dof_actuation[ self.six_dof_actuation_mask ] = actuation

        self.buoyancy[ 2 ] = self.vertical_multiplier * rho_eau * G * self.volume * (1. - 1. / (1. + exp(
                self.vertical_multiplier * 10. * (self.water_surface_depth - state[ 2 ]) - 2.
        )))

        hydrostatic_forces = zeros( 6 )
        hydrostatic_forces[ :3 ] = transform_matrix[ :3, :3 ].T @ (self.weight + self.buoyancy)
        hydrostatic_forces[ 3: ] = cross(
                self.center_of_mass, transform_matrix[ :3, :3 ].T @ self.weight
        ) + cross(
                self.center_of_volume, transform_matrix[ :3, :3 ].T @ self.buoyancy
        )

        xdot = zeros( state.shape )
        xdot[ :6 ] = transform_matrix @ state[ 6: ]
        xdot[ 6: ] = self.inverse_inertial_matrix @ (-self.hydrodynamic_matrix @ (
                state[ 6: ] - self.water_current) + hydrostatic_forces + six_dof_actuation + perturbation)

        return xdot

    def compute_error( self, actual: ndarray, target: ndarray ) -> ndarray:
        error = actual - target
        error[ :, :, self.orientation ] %= pi
        return error

    def get_body_to_world_transform( self, state: ndarray ) -> ndarray:
        phi, theta, psi = state[ self.orientation ]

        cPhi, sPhi = cos( phi ), sin( phi )
        cTheta, sTheta, tTheta = cos( theta ), sin( theta ), tan( theta )
        cPsi, sPsi = cos( psi ), sin( psi )

        matrix = eye( 6 )
        matrix[ 0, 0 ] = cPsi * cTheta
        matrix[ 0, 1 ] = -sPsi * cPhi + cPsi * sTheta * sPhi
        matrix[ 0, 2 ] = sPsi * sPhi + cPsi * sTheta * cPhi
        matrix[ 1, 0 ] = sPsi * cTheta
        matrix[ 1, 1 ] = cPsi * cPhi + sPsi * sTheta * sPhi
        matrix[ 1, 2 ] = -cPsi * sPhi + sPsi * sTheta * cPhi
        matrix[ 2, 0 ] = -sTheta
        matrix[ 2, 1 ] = cTheta * sPhi
        matrix[ 2, 2 ] = cTheta * cPhi
        matrix[ 3, 4 ] = sPhi * tTheta
        matrix[ 3, 5 ] = cPhi * tTheta
        matrix[ 4, 3 ] = 0
        matrix[ 4, 4 ] = cPhi
        matrix[ 4, 5 ] = -sPhi
        matrix[ 5, 3 ] = 0
        matrix[ 5, 4 ] = sPhi / cTheta
        matrix[ 5, 5 ] = cPhi / cTheta
        return matrix

    @staticmethod
    def build_inertial_matrix(
            mass: float, center_of_mass: ndarray, inertial_coefficients: list
    ) -> ndarray:
        """
        build the inertial matrix from the mass, center of mass and inertial coefficients

        Parameters
        ----------
        mass : float
        center_of_mass : ndarray
        inertial_coefficients : list

        Returns
        -------
        ndarray
            inertial matrix of shape (6, 6)
        """
        inertial_matrix = eye( 6 )
        for i in range( 3 ):
            inertial_matrix[ i, i ] = mass
            inertial_matrix[ i + 3, i + 3 ] = inertial_coefficients[ i ]
        inertial_matrix[ 0, 4 ] = mass * center_of_mass[ 2 ]
        inertial_matrix[ 0, 5 ] = - mass * center_of_mass[ 1 ]
        inertial_matrix[ 1, 3 ] = - mass * center_of_mass[ 2 ]
        inertial_matrix[ 1, 5 ] = mass * center_of_mass[ 0 ]
        inertial_matrix[ 2, 3 ] = mass * center_of_mass[ 1 ]
        inertial_matrix[ 2, 4 ] = - mass * center_of_mass[ 0 ]
        inertial_matrix[ 4, 0 ] = mass * center_of_mass[ 2 ]
        inertial_matrix[ 5, 0 ] = - mass * center_of_mass[ 1 ]
        inertial_matrix[ 3, 1 ] = - mass * center_of_mass[ 2 ]
        inertial_matrix[ 5, 1 ] = mass * center_of_mass[ 0 ]
        inertial_matrix[ 3, 2 ] = mass * center_of_mass[ 1 ]
        inertial_matrix[ 4, 2 ] = - mass * center_of_mass[ 0 ]
        inertial_matrix[ 3, 4 ] = - inertial_coefficients[ 3 ]
        inertial_matrix[ 3, 5 ] = - inertial_coefficients[ 4 ]
        inertial_matrix[ 4, 5 ] = - inertial_coefficients[ 5 ]
        inertial_matrix[ 4, 3 ] = - inertial_coefficients[ 3 ]
        inertial_matrix[ 5, 3 ] = - inertial_coefficients[ 4 ]
        inertial_matrix[ 5, 4 ] = - inertial_coefficients[ 5 ]

        return inertial_matrix


class BluerovXYZ( Bluerov ):
    """
    implementation of the Bluerov model **with reduced actuation capabilities**, based on the BlueROV model from Blue
    Robotics
    parameters of the model are based on the BlueROV2 Heavy configuration
    and are stored in the class as class variables

    Parameters
    ----------
    water_surface_depth: float
        depth of the water surface
    water_current: ndarray
        water current in the x, y, z directions
    reference_frame: str
        reference frame of the model, either 'NED' or 'ENU'

    Methods
    -------
    **\_\_call\_\_**( *ndarray*, *ndarray*, *ndarray* ) -> *ndarray*:
        evaluates the dynamics
    **compute_error**( *ndarray*, *ndarray* ) -> *ndarray*:
        computes the error between two trajectories according to the system definition
    
    Properties
    ----------
    **state_size**: *int*:
        size of the expected state vector
    **actuation_size**: *int*:
        size of the expected actuation vector
    **position**: *ndarray*:
        indices of the position inside the state vector
    **orientation**: *ndarray*:
        indices of the orientation inside the state vector
    **velocity**: *ndarray*:
        indices of the velocity inside the state vector
    **body_rates**: *ndarray*:
        indices of the body rates inside the state vector
    **linear_actuation**: *ndarray*:
        indices of the linear actuation inside the actuation vector
    **angular_actuation**: *ndarray*:
        indices of the angular actuation inside the actuation vector
    **six_dof_actuation_mask**: *ndarray*:
        relation between a full six degrees of freedom actuation and the actuation of the model
    """

    _actuation_size = 3

    _linear_actuation = r_[ slice( 0, 3 ) ]
    _angular_actuation = r_[ slice( 0, 0 ) ]

    _six_dof_actuation_mask = r_[ slice( 0, 3 ) ]


class BluerovXYZPsi( Bluerov ):
    """
    implementation of the Bluerov model **with reduced actuation capabilities**, based on the BlueROV model from Blue
    Robotics
    parameters of the model are based on the BlueROV2 Heavy configuration
    and are stored in the class as class variables

    Parameters
    ----------
    water_surface_depth: float
        depth of the water surface
    water_current: ndarray
        water current in the x, y, z directions
    reference_frame: str
        reference frame of the model, either 'NED' or 'ENU'

    Methods
    -------
    **\_\_call\_\_**( *ndarray*, *ndarray*, *ndarray* ) -> *ndarray*:
        evaluates the dynamics
    **compute_error**( *ndarray*, *ndarray* ) -> *ndarray*:
        computes the error between two trajectories according to the system definition
    
    Properties
    ----------
    **state_size**: *int*:
        size of the expected state vector
    **actuation_size**: *int*:
        size of the expected actuation vector
    **position**: *ndarray*:
        indices of the position inside the state vector
    **orientation**: *ndarray*:
        indices of the orientation inside the state vector
    **velocity**: *ndarray*:
        indices of the velocity inside the state vector
    **body_rates**: *ndarray*:
        indices of the body rates inside the state vector
    **linear_actuation**: *ndarray*:
        indices of the linear actuation inside the actuation vector
    **angular_actuation**: *ndarray*:
        indices of the angular actuation inside the actuation vector
    **six_dof_actuation_mask**: *ndarray*:
        relation between a full six degrees of freedom actuation and the actuation of the model
    """

    _actuation_size = 4

    _linear_actuation = r_[ slice( 0, 3 ) ]
    _angular_actuation = r_[ 3 ]

    _six_dof_actuation_mask = r_[ slice( 0, 3 ), 5 ]


class BluerovXZPsi( Bluerov ):
    """
    implementation of the Bluerov model **with reduced actuation capabilities**, based on the BlueROV model from Blue
    Robotics
    parameters of the model are based on the BlueROV2 Heavy configuration
    and are stored in the class as class variables

    Parameters
    ----------
    water_surface_depth: float
        depth of the water surface
    water_current: ndarray
        water current in the x, y, z directions
    reference_frame: str
        reference frame of the model, either 'NED' or 'ENU'

    Methods
    -------
    **\_\_call\_\_**( *ndarray*, *ndarray*, *ndarray* ) -> *ndarray*:
        evaluates the dynamics
    **compute_error**( *ndarray*, *ndarray* ) -> *ndarray*:
        computes the error between two trajectories according to the system definition
    
    Properties
    ----------
    **state_size**: *int*:
        size of the expected state vector
    **actuation_size**: *int*:
        size of the expected actuation vector
    **position**: *ndarray*:
        indices of the position inside the state vector
    **orientation**: *ndarray*:
        indices of the orientation inside the state vector
    **velocity**: *ndarray*:
        indices of the velocity inside the state vector
    **body_rates**: *ndarray*:
        indices of the body rates inside the state vector
    **linear_actuation**: *ndarray*:
        indices of the linear actuation inside the actuation vector
    **angular_actuation**: *ndarray*:
        indices of the angular actuation inside the actuation vector
    **six_dof_actuation_mask**: *ndarray*:
        relation between a full six degrees of freedom actuation and the actuation of the model
    """

    _actuation_size = 3

    _linear_actuation = r_[ slice( 0, 2 ) ]
    _angular_actuation = r_[ 2 ]

    six_dof_actuation_mask = r_[ 0, 2, 5 ]


class USV( Bluerov ):
    """
    implementation of the Bluerov model **with reduced actuation capabilities as to represent a surface vehicle**,
    based on the BlueROV model from Blue Robotics
    parameters of the model are based on the BlueROV2 Heavy configuration
    and are stored in the class as class variables

    Parameters
    ----------
    water_surface_depth: float
        depth of the water surface
    water_current: ndarray
        water current in the x, y, z directions
    reference_frame: str
        reference frame of the model, either 'NED' or 'ENU'

    Methods
    -------
    **\_\_call\_\_**( *ndarray*, *ndarray*, *ndarray* ) -> *ndarray*:
        evaluates the dynamics
    **compute_error**( *ndarray*, *ndarray* ) -> *ndarray*:
        computes the error between two trajectories according to the system definition
    
    Properties
    ----------
    **state_size**: *int*:
        size of the expected state vector
    **actuation_size**: *int*:
        size of the expected actuation vector
    **position**: *ndarray*:
        indices of the position inside the state vector
    **orientation**: *ndarray*:
        indices of the orientation inside the state vector
    **velocity**: *ndarray*:
        indices of the velocity inside the state vector
    **body_rates**: *ndarray*:
        indices of the body rates inside the state vector
    **linear_actuation**: *ndarray*:
        indices of the linear actuation inside the actuation vector
    **angular_actuation**: *ndarray*:
        indices of the angular actuation inside the actuation vector
    **six_dof_actuation_mask**: *ndarray*:
        relation between a full six degrees of freedom actuation and the actuation of the model
    """

    _actuation_size = 2

    _linear_actuation = r_[ 0 ]
    _angular_actuation = r_[ 1 ]

    _six_dof_actuation_mask = r_[ 0, 5 ]


if __name__ == '__main__':
    from numpy import set_printoptions
    from numpy.random import random

    set_printoptions( precision=2, linewidth=10000, suppress=True )

    for model in [ Bluerov, BluerovXYZ, BluerovXYZPsi, BluerovXZPsi, USV ]:
        print( model.__name__ )

        br = model()

        print( f"{br.state_size=}" )
        print( f"{br.actuation_size=}" )
        print( f"{br.position=}" )
        print( f"{br.orientation=}" )
        print( f"{br.velocity=}" )
        print( f"{br.body_rates=}" )
        print( f"{br.linear_actuation=}" )
        print( f"{br.angular_actuation=}" )
        print( f"{br.six_dof_actuation_mask=}" )

        s = random( (br.state_size,) )
        a = random( (br.actuation_size,) )
        p = random( (br.state_size // 2,) )
        ds = br( s, a, p )

        t = random( (10, 1, br.state_size // 2) )
        a = random( (10, 1, br.state_size // 2) )
        e = br.compute_error( a, t )
        print( f"{e=}" )
