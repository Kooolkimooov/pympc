from numpy import array, cross, ndarray
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from pympc.models.catenary import Catenary


class AugmentedCatenary( Catenary ):
    """
    Implementation of the augmented catenary model for a cable
    https://hal.science/hal-04459364/

    Parameters
    ----------
    length: float
        length of the cable
    linear_mass: float
        linear mass of the cable
    get_parameter_method: str
        method to get the parameters of the catenary, either 'runtime' or 'precompute'
    reference_frame: str
        reference frame of the cable, either 'NED' or 'ENU'

    Methods
    -------
        **\_\_call\_\_**( *ndarray*, *ndarray*, *float* = 0, *float* = 0 ) -> *tuple*:
            get all data from other methods in an optimized way
        **get_lowest_point**( *ndarray*, *ndarray*, *float* = 0, *float* = 0 ) -> *ndarray*:
            get the catenary's lowest point
        **get_perturbations**( *ndarray*, *ndarray*, *float* = 0, *float* = 0 ) -> *tuple*:
            get the perturbations of the catenary on the two points
        **discretize**( *ndarray*, *ndarray*, *float* = 0, *float* = 0, *int* = 100 ) -> *ndarray*:
            discretize the catenary; if the optimization fails, the catenary is approximated by a straight line
        **get_parameters**( *ndarray*, *ndarray*, *float* = 0, *float* = 0 ) -> *tuple*:
            get the parameters of the catenary

    Attributes
    ----------
    length: float
        length of the cable
    linear_mass: float
        linear mass of the cable
    """

    def __init__(
            self, length=3., linear_mass=1., get_parameter_method: str = 'runtime', reference_frame: str = 'NED'
    ):
        super().__init__( length, linear_mass, get_parameter_method, reference_frame )
        self._get_base_parameters = self.get_parameters
        self.get_parameters = self._get_augmented_parameters

    def __call__( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0. ) -> tuple:
        """
        compute all relevant data on the catenary with the current attributes, and the given
        attachment points

        Parameters
        ----------
        p0: ndarray
            first attachment point
        p1: ndarray
            second attachment point
        gamma: float
            tilt angle of the catenary plane around the p0-p1 axis; default is 0
        theta: float
            in plane tilt angle around the catenary plane's normal axis; default is 0

        Returns
        -------
        tuple:
            - **float**: the parameter C of the catenary; set to None if out of safe search space
            - **float**: the vertical sag H of the catenary; set to None if out of safe search space and 2D+ΔD > length
            - **float**: the vertical distance ΔH between attachment points; set to None if out of safe search space and 2D+ΔD > length
            - **float**: the horizontal half-length D; set to None if out of safe search space and 2D+ΔD > length
            - **float**: horizontal asymmetric length ΔD; set to None if out of safe search space and 2D+ΔD > length
        ndarray: 
            the lowest point (x, y, z) of the catenary; CURRENTLY UNRELIABLE
        tuple:
            - **ndarray**: the perturbations force on p0
            - **ndarray**: the perturbations force on p1
        ndarray: 
            the shape of the catenary with 100 curvilinearly discretized points (x, y, z) are on the second dimension of the array.
        """

        C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation = self.get_parameters( p0, p1, gamma, theta )

        lowest_point = self._get_lowest_point( p0, virtual_p1, C, H, dH, D, dD )
        lowest_point = self._augmented_rotation( lowest_point, p0, gamma_rotation, theta_rotation )

        perturbations = self._get_perturbations( p0, virtual_p1, C, H, dH, D, dD )
        perturbations = (
                self._augmented_rotation( perturbations[ 0 ], p0, gamma_rotation, theta_rotation ),
                self._augmented_rotation( perturbations[ 1 ], p0, gamma_rotation, theta_rotation )
        )

        points = self._discretize( p0, virtual_p1, C, H, D, dD )
        points = self._augmented_rotation( points, p0, gamma_rotation, theta_rotation )

        return (C, H, dH, D, dD), lowest_point, perturbations, points

    def get_lowest_point( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0. ) -> ndarray:
        """
        computes the catenary's lowest point

        this method is CURRENTLY UNRELIABLE with some augmented catenary configurations
        because it the lowest point is computed for the catenary with the virtual point
        before it is rotated back into the original frame.

        Parameters
        ----------
        p0:
            first attachment point
        p1:
            second attachment point
        gamma: float
            tilt angle of the catenary plane around the p0-p1 axis; default is 0
        theta: float
            in plane tilt angle around the catenary plane's normal axis; default is 0

        Returns
        -------
        ndarray:
            the lowest point (x, y, z) of the catenary
        """
        C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation = self.get_parameters( p0, p1, gamma, theta )

        lowest_point = self._get_lowest_point( p0, virtual_p1, C, H, dH, D, dD )
        lowest_point = self._augmented_rotation( lowest_point, p0, gamma_rotation, theta_rotation )

        return lowest_point

    def get_perturbations( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0. ) -> tuple:
        """
        computes the perturbations of the catenary on the two points

        Parameters
        ----------
        p0: ndarray
            one end of the catenary
        p1: ndarray
            second end of the catenary
        gamma: float
            tilt angle of the catenary plane around the p0-p1 axis; default is 0
        theta: float
            in plane tilt angle around the catenary plane's normal axis; default is 0

        Returns
        -------
        tuple: 
            the perturbations force on the two points in the form (perturbation_p1, 
            perturbation_p2)
"""
        C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation = self.get_parameters( p0, p1, gamma, theta )

        perturbations = self._get_perturbations( p0, virtual_p1, C, H, dH, D, dD )
        perturbations = (
                self._augmented_rotation( perturbations[ 0 ], p0, gamma_rotation, theta_rotation ),
                self._augmented_rotation( perturbations[ 1 ], p0, gamma_rotation, theta_rotation )
        )

        return perturbations

    def discretize( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0., n: int = 100 ) -> ndarray:
        """
        discretize the catenary, if the optimization fails, the catenary is approximated by a straight line

        Parameters
        ----------

        p0: ndarray
            one end of the catenary
        p1: ndarray
            second end of the catenary
        gamma: float
            tilt angle of the catenary plane around the p0-p1 axis; default is 0
        theta: float
            in plane tilt angle around the catenary plane's normal axis; default is 0
        n: int
            number of point to discretize; default is 100

        Returns
        -------
        ndarray:
            array of points of the catenary points are on the second dimension of the array
        """
        C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation = self.get_parameters( p0, p1, gamma, theta )

        points = self._discretize( p0, virtual_p1, C, H, D, dD, n )
        points = self._augmented_rotation( points, p0, gamma_rotation, theta_rotation )

        return points
    
    def get_parameters( self, p0: ndarray, p1: ndarray ) -> tuple:
        """
        computes the parameters of the catenary

        Parameters
        ----------
        p0: ndarray
            first attachment point
        p1: ndarray
            second attachment point
        gamma: float
            tilt angle of the catenary plane around the p0-p1 axis
        theta: float
            in plane tilt angle around the catenary plane normal axis

        Returns
        -------
        tuple:
            - **float**: the parameter C of the catenary; set to None if out of safe search space
            - **float**: the vertical sag H of the catenary; set to None if out of safe search space and 2D+ΔD > length
            - **float**: the vertical distance ΔH between attachment points; set to None if out of safe search space and 2D+ΔD > length
            - **float**: the horizontal half-length D; set to None if out of safe search space and 2D+ΔD > length
            - **float**: horizontal asymmetric length ΔD; set to None if out of safe search space and 2D+ΔD > length
        """
        raise NotImplementedError( 'get_parameters method should have been implemented in __init__' )

    def _get_augmented_parameters( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0. ) -> tuple:
        gamma_axis = p0 - p1
        gamma_axis /= norm( gamma_axis )
        gamma_rotation = Rotation.from_rotvec( self.vertical_multiplier * gamma_axis * gamma ).as_matrix()

        theta_axis = cross( array( [ 0., 0., 1. ] ), gamma_axis )
        theta_rotation = Rotation.from_rotvec( self.vertical_multiplier * theta_axis * theta ).as_matrix()

        virtual_p1 = (p1 - p0) @ theta_rotation + p0

        C, H, dH, D, dD = self._get_base_parameters( p0, virtual_p1 )
        return C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation

    @staticmethod
    def _augmented_rotation(
            p: ndarray,
            p0: ndarray,
            gamma_rotation: ndarray,
            theta_rotation: ndarray
            ) -> ndarray:
        return (p - p0) @ theta_rotation.T @ gamma_rotation + p0
