from time import perf_counter

from numpy import arctan2, cos, cosh, eye, ndarray, sin, sinh, sqrt, zeros
from numpy.linalg import pinv

from pympc.models.catenary import Catenary


class VS:
    """
    implements a pseudo visual servoing controller for tethered robots.
    DOI: 10.1109/ICRA.2017.7989090,
    https://www.researchgate.net/publication/317740842_Catenary-based_Visual_Servoing_for_Tethered_Robots

    Parameters
    ----------

    leader_pose: ndarray
        the pose of the leader
    follower_pose: ndarray
        the pose of the leader
    target_feature: ndarray
        the target feature in the form [H / H_max, sin(α)] with alpha the angle
        between the orientation of the robot and the orientation of the catenary
        projected in the horizontal (x-y) plane
    cable_length: float
        the length of the cable modelled by the catenary
    maximum_H: float
        the maximum allowed value of the catenary sag H
    actuation_projection_matrix: ndarray
        the projection matrix from the velocity vector to the actuation degrees of actuation
    record: bool
        whether to record the computation times
    verbose: bool
        whether to print the optimization results

    Methods
    -------
    **compute_actuation**():
        computes the best actuation for the current state of the model
    **compute_result**():
        computes an actionable actuation from `raw_result` depending on the configuration

    Properties
    ----------
    **target_trajectory**: *ndarray*:
        target trajectory
    **record**: *bool*:
        whether to record the computation times, predicted trajectories and candidate actuations
    **verbose**: *bool*:
        whether to print the optimization results
    **compute_times**: *list*:
        list of computation times
    **result**: *ndarray*:
        best actuation found during the optimization
    **raw_result**: *OptimizeResult*:
        raw result of the optimization
    """

    REFERENCE_FRAME = [ 'NED', 'ENU' ]

    def __init__(
            self,
            leader_pose: ndarray,
            follower_pose: ndarray,
            target_feature: ndarray,
            cable_length: float = 1.0,
            maximum_H: float = None,
            gain: float = 1.0,
            actuation_projection_matrix: ndarray = None,
            reference_frame: str = 'NED',
            record: bool = False,
            verbose: bool = False
    ):
        if reference_frame not in self.REFERENCE_FRAME:
            raise ValueError( f'reference_frame must be one of {self.REFERENCE_FRAME}' )

        if leader_pose.flatten().shape[ 0 ] != 6 or follower_pose.flatten().shape[ 0 ] != 6:
            raise ValueError( 'pose should be of size 6' )

        if maximum_H is not None and (maximum_H <= 0 or maximum_H > cable_length / 2):
            raise ValueError( 'maximum_H must be lower than half the cable length and positive' )

        if gain <= 0:
            raise ValueError( 'gain must be positive' )

        self.leader_pose = leader_pose
        self.follower_pose = follower_pose

        self.target_feature = target_feature
        self.catenary = Catenary(
                length=cable_length, linear_mass=0, get_parameter_method='runtime', reference_frame=reference_frame
        )

        if actuation_projection_matrix is None:
            self.actuation_projection_matrix = eye( 6 )

        self.raw_result = zeros( (6,) )
        self.result = zeros( (self.actuation_projection_matrix.shape[ 0 ],) )

        if maximum_H is None:
            self.maximum_H = self.catenary.length / 2

        self.gain = gain

        self.compute_times = [ ]

        self.record = record
        self.verbose = verbose

    def compute_actuation( self ) -> ndarray:
        """
        computes the best actuation for the current state with a given horizon. records the computation
        time if record is True and returns the best actuation

        Returns
        -------
        ndarray:
            best actuation for the next step; shape = (actuation_size,)
        """

        if self.record:
            ti = perf_counter()

        follower_angle = float( self.follower_pose[ 5 ] )
        dx = float( self.leader_pose[ 0 ] - self.follower_pose[ 0 ] )
        dy = float( self.leader_pose[ 1 ] - self.follower_pose[ 1 ] )
        cable_angle = arctan2( dy, dx )
        delta = cable_angle - follower_angle
        angle = arctan2( sin( delta ), cos( delta ) )

        C, H, dH, D, dD = self.catenary.get_parameters( self.leader_pose[ :3 ], self.follower_pose[ :3 ] )

        feature = ndarray( [ H / self.maximum_H, sin( angle ) ] )

        R = sinh( (D + dD) * C ) / C  # catenary half length
        Kc = 2 * (pow( R, 2 ) + pow( H, 2 )) / pow( pow( R, 2 ) - pow( H, 2 ), 2 )
        Kh = sinh( C * D ) / (1 + Kc * (cosh( C * D ) - 1 - C * D * sinh( C * D )) / pow( C, 2 ))

        interaction_matrix = zeros( (6, 2) )
        interaction_matrix[ 0, 0 ] = -Kh * sqrt( 1 - pow( feature[ 1 ], 2 ) ) / (2 * self.maximum_H)
        interaction_matrix[ 1, 0 ] = -Kh * feature[ 1 ] / (2 * self.maximum_H)
        interaction_matrix[ 5, 0 ] = Kh * (
                self.leader_pose[ 1 ] * sqrt( 1 - pow( feature[ 1 ], 2 ) ) - self.leader_pose[ 0 ] * feature[ 1 ]) / (
                                             2 * self.maximum_H)
        interaction_matrix[ 0, 1 ] = -feature[ 1 ] * sqrt( 1 - pow( feature[ 1 ], 2 ) ) / (2 * D)
        interaction_matrix[ 0, 2 ] = (-1 + pow( feature[ 1 ], 2 )) / (2 * D)
        interaction_matrix[ 5, 1 ] = - (
                self.leader_pose[ 1 ] * feature[ 1 ] * sqrt( 1 - pow( feature[ 1 ], 2 ) ) + self.leader_pose[
            0 ] * (1 - pow( feature[ 1 ], 2 ))) / (2 * D)

        self.raw_result = -self.gain * pinv( interaction_matrix ) @ (feature - self.target_feature)

        if self.record:
            self.compute_times.append( perf_counter() - ti )

        self.compute_result()

        return self.result

    def compute_result( self ):
        """
        computes the best actuation from scipy.optimize raw result and store it in self.result
        """
        self.result = self.actuation_projection_matrix @ self.raw_result
