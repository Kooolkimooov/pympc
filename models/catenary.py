from json import dump, load
from pathlib import Path

from numpy import (
    arccosh, arcsinh, array, cosh, isnan, linspace, log10, logspace, meshgrid, ndarray, sinh, sqrt, zeros,
)
from numpy.linalg import norm
from scipy.optimize import brentq
from tqdm import tqdm

from pympc.utils import G, check


class Catenary:
    """
    implementation of the catenary model for a cable

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
        **\_\_call\_\_**( *ndarray*, *ndarray* ) -> *tuple*:
            get all data from other methods in an optimized way
        **get_lowest_point**( *ndarray*, *ndarray* ) -> *ndarray*:
            get the catenary's lowest point
        **get_perturbations**( *ndarray*, *ndarray* ) -> *tuple*:
            get the perturbations of the catenary on the two points
        **discretize**( *ndarray*, *ndarray*, *int* = 100 ) -> *ndarray*:
            discretize the catenary; if the optimization fails, the catenary is approximated by a straight line
        **get_parameters**( *ndarray*, *ndarray* ) -> *tuple*:
            get the parameters of the catenary

    Attributes
    ----------
    length: float
        length of the cable
    linear_mass: float
        linear mass of the cable
    """

    GET_PARAMETER_METHOD = [ 'runtime', 'precompute' ]
    REFERENCE_FRAME = [ 'NED', 'ENU' ]

    def __init__(
            self, length=3., linear_mass=1., get_parameter_method: str = 'runtime', reference_frame: str = 'NED'
    ):
        self.length = length
        self.linear_mass = linear_mass
        self.optimization_function = self._optimization_function_0

        if reference_frame == 'NED':
            self.vertical_multiplier = -1.
        elif reference_frame == 'ENU':
            self.vertical_multiplier = 1.
        else:
            raise ValueError( f'reference_frame must be one of {self.REFERENCE_FRAME}' )

        if get_parameter_method == 'runtime':
            self.get_parameters = self._get_parameters_runtime
        elif get_parameter_method == 'precompute':
            self._precompute()
            self.get_parameters = self._get_parameters_precompute
        else:
            raise ValueError( f'get_parameter_method must be one of {self.GET_PARAMETER_METHOD}' )

    def __call__( self, p0: ndarray, p1: ndarray ) -> tuple:
        """
        compute all relevant data on the catenary with the current attributes, and the given
        attachment points

        Parameters
        ----------
        p0: ndarray
            first attachment point
        p1: ndarray
            second attachment point

        Returns
        -------
        tuple:
            - **float**: the parameter C of the catenary; set to None if out of safe search space
            - **float**: the vertical sag H of the catenary; set to None if out of safe search space and 2D+ΔD > length
            - **float**: the vertical distance ΔH between attachment points; set to None if out of safe search space
            and 2D+ΔD > length
            - **float**: the horizontal half-length D; set to None if out of safe search space and 2D+ΔD > length
            - **float**: horizontal asymmetric length ΔD; set to None if out of safe search space and 2D+ΔD > length
        ndarray: 
            the lowest point (x, y, z) of the catenary
        tuple:
            - **ndarray**: the perturbations force on p0
            - **ndarray**: the perturbations force on p1
        ndarray: 
            the shape of the catenary with 100 curvilinearly discretized points (x, y, z) are on the second dimension
            of the array.
        """

        C, H, dH, D, dD = self.get_parameters( p0, p1 )
        lowest_point = self._get_lowest_point( p0, p1, C, H, dH, D, dD )
        perturbations = self._get_perturbations( p0, p1, C, H, dH, D, dD )
        points = self._discretize( p0, p1, C, H, D, dD )

        return (C, H, dH, D, dD), lowest_point, perturbations, points

    def get_lowest_point( self, p0: ndarray, p1: ndarray ) -> ndarray:
        """
        computes the catenary's lowest point

        Parameters
        ----------
        p0:
            first attachment point
        p1:
            second attachment point

        Returns
        -------
        ndarray:
            the lowest point (x, y, z) of the catenary
        """
        C, H, dH, D, dD = self.get_parameters( p0, p1 )
        return self._get_lowest_point( p0, p1, C, H, dH, D, dD )

    def get_perturbations( self, p0: ndarray, p1: ndarray ) -> tuple:
        """
        computes the perturbations of the catenary on the two points

        Parameters
        ----------
        p0: ndarray
            one end of the catenary
        p1: ndarray
            second end of the catenary

        Returns
        -------
        tuple: 
            the perturbations force on the two points in the form (perturbation_p1, 
            perturbation_p2)
        """
        C, H, dH, D, dD = self.get_parameters( p0, p1 )
        return self._get_perturbations( p0, p1, C, H, dH, D, dD )

    def discretize( self, p0: ndarray, p1: ndarray, n: int = 100 ) -> ndarray:
        """
        discretize the catenary, if the optimization fails, the catenary is approximated by a straight line

        Parameters
        ----------

        p0: ndarray
            one end of the catenary
        p1: ndarray
            second end of the catenary
        n: int
            number of point to discretize; default is 100

        Returns
        -------
        ndarray:
            array of points of the catenary points are on the second dimension of the array
        """
        C, H, _, D, dD = self.get_parameters( p0, p1 )
        return self._discretize( p0, p1, C, H, D, dD, n )

    def get_parameters( self, p0: ndarray, p1: ndarray ) -> tuple:
        """
        computes the parameters of the catenary

        Parameters
        ----------
        p0: ndarray
            first attachment point
        p1: ndarray
            second attachment point

        Returns
        -------
        tuple:
            - **float**: the parameter C of the catenary; set to None if out of safe search space
            - **float**: the vertical sag H of the catenary; set to None if out of safe search space and 2D+ΔD > length
            - **float**: the vertical distance ΔH between attachment points; set to None if out of safe search space
            and 2D+ΔD > length
            - **float**: the horizontal half-length D; set to None if out of safe search space and 2D+ΔD > length
            - **float**: horizontal asymmetric length ΔD; set to None if out of safe search space and 2D+ΔD > length
        """
        raise NotImplementedError( 'get_parameters method should have been implemented in __init__' )

    @staticmethod
    def optimization_function( C, length, dH, two_D_plus_dD ):
        raise NotImplementedError( 'optimization_function method should have been implemented in __init__' )

    def _get_parameters_runtime( self, p0: ndarray, p1: ndarray ) -> tuple:
        dH = self.vertical_multiplier * (p0[ 2 ] - p1[ 2 ])
        two_D_plus_dD = norm( p1[ :2 ] - p0[ :2 ] )

        if norm( p1 - p0 ) > 0.99 * self.length or any( isnan( p0 ) ) or any( isnan( p1 ) ):
            return None, None, dH, None, None
        elif two_D_plus_dD < .01 * self.length:
            return None, (self.length - dH) / 2, dH, two_D_plus_dD / 2, 0.

        C = brentq(
                self.optimization_function, -1e-2, 1e3, args=(self.length, dH, two_D_plus_dD), xtol=1e-12
        )

        temp_var = pow( self.length, 2 ) - pow( dH, 2 )

        a_eq = -4. * pow( C, 2 ) * temp_var
        b_eq = -4. * C * dH * (C * temp_var - 2 * dH) - 8.0 * pow( self.length, 2 ) * C
        c_eq = pow( C * temp_var - 2. * dH, 2 )

        H = (-b_eq - sqrt( pow( b_eq, 2 ) - 4. * a_eq * c_eq )) / (2. * a_eq)
        D = arccosh( C * H + 1.0 ) / C
        dD = two_D_plus_dD - 2. * D

        return C, H, dH, D, dD

    def _precompute( self ):
        self._dHs = linspace( 0., self.length, 1000 )
        self._two_D_plus_dDs = self.length * logspace( -2, 0, 1000 )

        check( Path( f'./cache' ), prompt=False )
        check( Path( f'./cache/Catenary' ), prompt=False )
        if len( list( Path( f'./cache/Catenary' ).glob( f'{self.length}*' ) ) ):
            with open( Path( f'./cache/Catenary/{self.length}.json' ) ) as file:
                self._Cs = array( load( file ) )
                return

        X, Z = meshgrid( self._two_D_plus_dDs, self._dHs )
        self._Cs = zeros( X.shape )

        for i, xr in enumerate( tqdm( X, desc='precomputing values of C' ) ):
            for j, x in enumerate( xr ):
                z = Z[ i, j ]
                p1 = array( [ 0., 0., 0. ] )
                p2 = array( [ x, 0., z ] )
                self._Cs[ i, j ], _, _, _, _ = self._get_parameters_runtime( p1, p2 )

        with open( Path( f'./cache/Catenary/{self.length}.json' ), 'w' ) as file:
            dump( self._Cs.tolist(), file )

    def _get_parameters_precompute( self, p0: ndarray, p1: ndarray ) -> tuple:
        dH = self.vertical_multiplier * (p0[ 2 ] - p1[ 2 ])
        two_D_plus_dD = norm( p1[ :2 ] - p0[ :2 ] )

        if norm( p1 - p0 ) > 0.99 * self.length or any( isnan( p0 ) ) or any( isnan( p1 ) ):
            return None, None, dH, None, None
        elif two_D_plus_dD < .01 * self.length:
            return None, (self.length - dH) / 2, dH, two_D_plus_dD / 2, 0.

        i = int( round( (1000 - 1) * abs( dH ) / self.length, 0 ) )
        j = int( round( (1000 - 1) * (log10( abs( two_D_plus_dD ) / self.length ) - (-2)) / (0 - (-2)), 0 ) )
        if (0 < i and not self._dHs[ i - 1 ] < abs( dH )) or (i < 999 and not abs( dH ) < self._dHs[ i + 1 ]):
            raise ValueError()
        if (0 < j and not self._two_D_plus_dDs[ j - 1 ] < abs( two_D_plus_dD )) or (
                j < 999 and not abs( two_D_plus_dD ) < self._two_D_plus_dDs[ j + 1 ]):
            raise ValueError()

        C = self._Cs[ i, j ]

        if isnan( C ):
            return None, None, dH, None, None

        temp_var = pow( self.length, 2 ) - pow( dH, 2 )

        a_eq = -4. * pow( C, 2 ) * temp_var
        b_eq = -4. * C * dH * (C * temp_var - 2 * dH) - 8.0 * pow( self.length, 2 ) * C
        c_eq = pow( C * temp_var - 2. * dH, 2 )

        H = (-b_eq - sqrt( pow( b_eq, 2 ) - 4. * a_eq * c_eq )) / (2. * a_eq)
        D = arccosh( C * H + 1.0 ) / C
        dD = two_D_plus_dD - 2. * D

        return C, H, dH, D, dD

    def _get_lowest_point(
            self, p0: ndarray, p1: ndarray, C: float, H: float, dH: float, D: float, dD: float
    ) -> ndarray:
        # case where horizontal distance is too small
        if (C is None) and (H is not None):
            return p0 + array( [ 0, 0, H + dH ] )
        # case where cable is taunt
        elif C is None:
            return p0 if p0[ 2 ] >= p1[ 2 ] else p1

        lowest_point = zeros( (3,) )
        lowest_point[ :2 ] = p0[ :2 ] + (D + dD) * (p1[ :2 ] - p0[ :2 ]) / norm( p1[ :2 ] - p0[ :2 ] )
        lowest_point[ 2 ] = p0[ 2 ] - self.vertical_multiplier * (H + dH)
        return lowest_point

    def _get_perturbations(
            self, p0: ndarray, p1: ndarray, C: float, H: float, dH: float, D: float, dD: float
    ) -> tuple:
        # case where horizontal distance is too small
        if (C is None) and (D is not None):
            return array(
                    [ 0., 0., self.linear_mass * G * (H + dH) ]
            ), array(
                    [ 0., 0., self.linear_mass * G * H ]
            )
        # case where cable is taunt
        elif C is None:
            return None, None

        horizontal_perturbation = self.linear_mass * G / C
        vertical_perturbation_0 = horizontal_perturbation * sinh( -C * (D + dD) )
        vertical_perturbation_1 = horizontal_perturbation * sinh( C * D )

        direction = (p1[ :2 ] - p0[ :2 ]) / norm( p1[ :2 ] - p0[ :2 ] )

        perturbation_p0, perturbation_p1 = zeros( (3,) ), zeros( (3,) )
        perturbation_p0[ :2 ] = direction * horizontal_perturbation
        perturbation_p0[ 2 ] = self.vertical_multiplier * vertical_perturbation_0
        perturbation_p1[ :2 ] = -direction * horizontal_perturbation
        perturbation_p1[ 2 ] = - self.vertical_multiplier * vertical_perturbation_1

        return perturbation_p0, perturbation_p1

    def _discretize( self, p0: ndarray, p1: ndarray, C: float, H: float, D: float, dD: float, n: int = 100 ) -> ndarray:
        # case where ΔH is too small
        if (C is None) and (D is not None):
            return array( [ p0, p0 + array( [ 0, 0, H ] ), p1 ] )
        # case where cable is taunt
        elif C is None:
            return array( [ p0, p1 ] )

        points = zeros( (100, 3) )

        s = 0.
        ds = self.length / n

        for i in range( n - 1 ):
            # get x,z coord of points in catenary frame, centered at 1st point
            inter = C * s - sinh( C * D )
            x = 1. / C * arcsinh( inter ) + D
            z = 1. / C * (sqrt( 1. + pow( inter, 2 ) ) - 1.) - H
            points[ i, 0 ] = p1[ 0 ] - x * (p1[ 0 ] - p0[ 0 ]) / (2. * D + dD)
            points[ i, 1 ] = p1[ 1 ] - x * (p1[ 1 ] - p0[ 1 ]) / (2. * D + dD)
            points[ i, 2 ] = p1[ 2 ] + self.vertical_multiplier * z
            s += ds

        points[ -1 ] = p0

        return points

    @staticmethod
    def _optimization_function_0( C, length, dH, two_D_plus_dD ) -> float:
        return C * C * (length * length - dH * dH) - 2.0 * (-1.0 + cosh( C * two_D_plus_dD ))

    @staticmethod
    def _optimization_function_1( C, length, dH, two_D_plus_dD ) -> float:
        return pow( length, 2 ) - pow( dH, 2 ) - pow( 2 * sinh( C * two_D_plus_dD / 2 ) / C, 2 )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cat = Catenary( length=3.0, get_parameter_method="precompute" )

    # X, Z = meshgrid( cat._two_D_plus_dDs, cat._dHs )
    # Y = cat._Cs
    #
    # ax3d = plt.subplot( 111, projection='3d' )
    # ax3d.plot_surface(X, Z, Y, cmap='viridis', edgecolor='none')
    # plt.show()

    X = linspace( -2.5, 2.5, 10 )
    Z = linspace( -2.5, 2.5, 10 )

    for x in X:
        for z in Z:
            p1 = array( [ 0., 0., 0. ] )
            p2 = array( [ x, 0., z ] )

            P = cat.discretize( p1, p2 )

            ax3d = plt.subplot( 111, projection='3d' )
            ax3d.plot( *P.T )
            plt.title( f"{p2=}" )
            plt.show()
