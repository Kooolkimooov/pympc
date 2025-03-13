from inspect import signature

from numpy import array, ndarray


def seafloor_function_0( x, y ):
  from numpy import sin, exp
  z = 4.5
  z += 1. * sin( y / 4 )
  z += .5 * sin( x / 3 )
  # peak at (-3, 0)
  z -= 2.5 * exp( -8 * (pow( (x - (-3)), 2 ) + pow( (y - 0), 2 )) )
  return z


class Seafloor:
  def get_distance_to_seafloor( self, point: ndarray ) -> float:
    raise NotImplementedError()

  def get_seafloor_depth( self, point: ndarray ) -> float:
    raise NotImplementedError()


class SeafloorFromFunction( Seafloor ):
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
