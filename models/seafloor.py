from inspect import signature

from numpy import ndarray
from numpy import sin, exp, asarray


def seafloor_function_0( x, y ):
    x = asarray( x )
    y = asarray( y )
    z = 4.5
    z += 1. * sin( y / 4 )
    z += .5 * sin( x / 3 )
    # peak at (-3, 0)
    z -= 2.5 * exp( -8 * (pow( (x - (-3)), 2 ) + pow( (y - 0), 2 )) )
    return z


class Seafloor:
    """
    abstract class for the seafloor

    Methods
    -------
    **get_distance_to_seafloor**( *ndarray* ) -> *float*:
        get the vertical distance to the seafloor from a point in 3D space
    **get_seafloor_depth**( *ndarray* ) -> *float*:
        get the depth of the seafloor at a point on the x-y plane
        if the point is 3d, the z coordinate is ignored
    """

    def get_distance_to_seafloor( self, point: ndarray ) -> float:
        """
        get the vertical distance to the seafloor from a point in 3D space

        Parameters
        ----------
        point : ndarray
            point in 3D space with shape (3,)

        Returns
        -------
        float
            vertical distance to the seafloor from the point
        """
        raise NotImplementedError()

    def get_seafloor_depth( self, point: ndarray ) -> float:
        """
        get the depth of the seafloor at a point on the x-y plane
        if the point is 3d, the z coordinate is ignored

        Parameters
        ----------
        point : ndarray
            point on the x-y plane 

        Returns
        -------
        float
            depth of the seafloor at the point
        """
        raise NotImplementedError()


class SeafloorFromFunction( Seafloor ):
    """
    implementation of the seafloor class that uses a function to define the seafloor

    Parameters
    ----------
    seafloor : callable
        a function that takes two arguments (x, y) and returns the depth of the seafloor

    Methods
    -------
    **get_distance_to_seafloor**( *ndarray* ) -> *float*:
        get the vertical distance to the seafloor from a point in 3D space
    **get_seafloor_depth**( *ndarray* ) -> *float*:
        get the depth of the seafloor at a point on the x-y plane
        if the point is 3d, the z coordinate is ignored

    Attributes
    ----------
    seafloor_function : callable
        a function that takes two arguments (x, y) and returns the depth of the seafloor
    """

    def __init__( self, seafloor: callable ):
        assert list( signature( seafloor ).parameters ) == [ 'x', 'y' ], 'provided function has irregular signature'
        self.seafloor_function = seafloor

    def get_distance_to_seafloor( self, point: ndarray ) -> float:
        return self.seafloor_function( *(point[ :2 ]) ) - point[ 2 ]

    def get_seafloor_depth( self, point: ndarray ) -> float:
        return self.seafloor_function( *(point[ :2 ]) )


class SeafloorFromArray( Seafloor ):
    def __init__( self, seafloor: ndarray ):
        raise NotImplementedError()
