from numpy import array, cross, ndarray
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from pympc.models.catenary import Catenary


class AugmentedCatenary( Catenary ):
  """
  Implementation of the augmented catenary model for a cable
  https://hal.science/hal-04459364/
  """

  def __init__(
      self, length = 3., linear_mass = 1., get_parameter_method: str = 'runtime', reference_frame: str = 'NED'
      ):
    """
    :param length: length of the cable
    :param linear_mass: linear mass of the cable
    :param get_parameter_method: method to get the parameters of the catenary (runtime or precompute)
    :param reference_frame: reference frame of the cable, either 'NED' or 'ENU'
    """
    super().__init__( length, linear_mass, get_parameter_method, reference_frame )
    self._get_base_parameters = self.get_parameters
    self.get_parameters = self._get_augmented_parameters

  def __call__( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0. ) -> tuple:
    """
    get all relevant data on the catenary of length self.length, linear mass self.linear_mass, and the given
    attachment points

    :param p0: first attachment point
    :param p1: second attachment point
    :param gamma: tilt angle of the catenary plane around the p0-p1 axis
    :param theta: in plane tilt angle around the catenary plane y-axis
    :return: tuple containing:
    - the parameters of the catenary:
      - the parameter of the catenary (C, set to None if out of safe search space);
      - vertical sag of the catenary (H, set to None if out of safe search space and 2D+ΔD > length);
      - vertical distance between attachment points (ΔH, set to None if out of safe search space and 2D+ΔD >
      length);
      - horizontal half-length (D, set to None if out of safe search space and 2D+ΔD > length);
      - horizontal asymmetric length (ΔD, set to None if out of safe search space and 2D+ΔD > length);
    - the lowest point (x, y, z) of the catenary;
    - the perturbations force on the two points in the form (perturbation_p0, perturbation_p1);
    - array of points for plotting (x, y, z) are on the second dimension of the array.
    """

    C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation = self.get_parameters( p0, p1, gamma, theta )

    lowest_point = self._get_lowest_point( p0, virtual_p1, C, H, dH, D, dD )
    lowest_point = self._augmented_rotation( lowest_point, p0, gamma_rotation, theta_rotation )

    perturbations = self._get_perturbations( p0, virtual_p1, C, H, dH, D, dD )
    perturbations = (self._augmented_rotation( perturbations[ 0 ], p0, gamma_rotation, theta_rotation ),
                     self._augmented_rotation( perturbations[ 1 ], p0, gamma_rotation, theta_rotation ))

    points = self._discretize( p0, virtual_p1, C, H, D, dD )
    points = self._augmented_rotation( points, p0, gamma_rotation, theta_rotation )

    return (C, H, dH, D, dD), lowest_point, perturbations, points

  def get_lowest_point( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0. ) -> ndarray:
    """
    get the catenary's lowest point

    :param p0: one end of the catenary
    :param p1: second end of the catenary
    :param gamma: tilt angle of the catenary plane around the p0-p1 axis
    :param theta: in plane tilt angle around the catenary plane y-axis
    :return: the lowest point (x, y, z) of the catenary
    """
    C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation = self.get_parameters( p0, p1, gamma, theta )

    lowest_point = self._get_lowest_point( p0, virtual_p1, C, H, dH, D, dD )
    lowest_point = self._augmented_rotation( lowest_point, p0, gamma_rotation, theta_rotation )

    return lowest_point

  def get_perturbations( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0. ) -> tuple:
    """
    get the perturbations of the catenary on the two points

    :param p0: one end of the catenary
    :param p1: second end of the catenary
    :param gamma: tilt angle of the catenary plane around the p0-p1 axis
    :param theta: in plane tilt angle around the catenary plane y-axis
    :return: tuple containing the perturbations force on the two points in the form (perturbation_p1,
    perturbation_p2)
    """
    C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation = self.get_parameters( p0, p1, gamma, theta )

    perturbations = self._get_perturbations( p0, virtual_p1, C, H, dH, D, dD )
    perturbations = (self._augmented_rotation( perturbations[ 0 ], p0, gamma_rotation, theta_rotation ),
                     self._augmented_rotation( perturbations[ 1 ], p0, gamma_rotation, theta_rotation ))

    return perturbations

  def discretize( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0., n: int = 100 ) -> ndarray:
    """
    discretize the catenary, if the optimization fails, the catenary is approximated by a straight line

    :param p0: one end of the catenary
    :param p1: second end of the catenary
    :param gamma: tilt angle of the catenary plane around the p0-p1 axis
    :param theta: in plane tilt angle around the catenary plane y-axis
    :param n: number of point to discretize
    :return: array of points of the catenary points are on the second dimension of the array
    """
    C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation = self.get_parameters( p0, p1, gamma, theta )

    points = self._discretize( p0, virtual_p1, C, H, D, dD, n )
    points = self._augmented_rotation( points, p0, gamma_rotation, theta_rotation )

    return points

  def _get_augmented_parameters( self, p0: ndarray, p1: ndarray, gamma: float = 0., theta: float = 0. ) -> tuple:
    """
    :param p0: first attachment point
    :param p1: second attachment point
    :param gamma: tilt angle of the catenary plane around the p0-p1 axis
    :param theta: in plane tilt angle around the catenary plane y-axis
    :return: tuple containing:
    - the parameter of the catenary (C, set to None if out of safe search space);
    - vertical sag of the catenary (H, set to None if out of safe search space and 2D+ΔD > length);
    - vertical distance between attachment points (ΔH, set to None if out of safe search space and 2D+ΔD > length);
    - horizontal half-length (D, set to None if out of safe search space and 2D+ΔD > length);
    - horizontal asymmetric length (ΔD, set to None if out of safe search space and 2D+ΔD > length)
    - the virtual second attachment point used for theta tilt
    - the gamma transformation matrix
    - the theta transformation matrix
    """

    gamma_axis = p0 - p1
    gamma_axis /= norm( gamma_axis )
    gamma_rotation = Rotation.from_rotvec( self.vertical_multiplier * gamma_axis * gamma ).as_matrix()

    theta_axis = cross( array( [ 0., 0., 1. ] ), gamma_axis )
    theta_rotation = Rotation.from_rotvec( self.vertical_multiplier * theta_axis * theta ).as_matrix()

    virtual_p1 = (p1 - p0) @ theta_rotation + p0

    C, H, dH, D, dD = self._get_base_parameters( p0, virtual_p1 )
    return C, H, dH, D, dD, virtual_p1, gamma_rotation, theta_rotation

  def _augmented_rotation( self, p: ndarray, p0: ndarray, gamma_rotation: ndarray, theta_rotation: ndarray ) -> ndarray:
    return (p - p0) @ theta_rotation.T @ gamma_rotation + p0
