from time import perf_counter

from numpy import arctan2, cos, cosh, eye, ndarray, sin, sinh, sqrt, zeros, cross, identity, array
from numpy.linalg import pinv
from scipy.spatial.transform import Rotation

from pympc.models.catenary import Catenary


class VS:
    """
    implements a pseudo visual servoing controller for tethered robots.
    Based on DOI: 10.1109/ICRA.2017.7989090,
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

    Properties
    ----------
    **leader_pose**: *ndarray*:
        the pose of the leader
    **follower_pose**: *ndarray*:
        the pose of the follower
    **target_feature**: *ndarray*:
        target feature in the form [H / H_max, sin(α)] with alpha the angle
        between the orientation of the robot and the orientation of the catenary
    **catenary**: *Catenary*:
        the catenary model used to model the cable
    **gain**: *float*:
        gain of the controller
    **maximum_H**: *float*:
        maximum allowed value of the catenary sag H
    **record**: *bool*:
        whether to record the computation times, predicted trajectories and candidate actuations
    **verbose**: *bool*:
        whether to print the optimization results
    **compute_times**: *list*:
        list of computation times
    **result**: *ndarray*:
        best actuation found during the optimization
    **raw_result**: *ndarray*:
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
            maximum_dH: float = None,
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
        
        if target_feature.flatten().shape[ 0 ] != 3:
            raise ValueError( 'target_feature should be of size 3' )

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
            self.actuation_projection_matrix = eye(6)

        self.raw_result = zeros( (6,) )
        self.result = zeros( (self.actuation_projection_matrix.shape[ 0 ],) )

        if maximum_H is None:
            self.maximum_H = self.catenary.length / 2
        else:
            self.maximum_H = maximum_H

        if maximum_dH is None:
            self.maximum_dH = self.catenary.length / 2
        else:
            self.maximum_dH = maximum_dH

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
        L = self.catenary.length

        a = H / self.maximum_H
        b = sin( angle )
        c = ( dH + self.maximum_dH ) / ( 2 * self.maximum_dH )

        feature = array( [ a, b, c ] )

        p = array([2 * D + dD, 0, dH]) @ Rotation.from_euler('z', follower_angle).as_matrix().T

        T = zeros( (3, 6) )
        T[:3, :3] = eye( 3 )
        T[:3, 3:] = -cross(identity(p.shape[0]) * -1, p)
        T *= -1

        LD = H - D * sinh( C * D )
        LdD = H + dH - ( D + dD ) * sinh( C * ( D + dD ) )

        Cn = 2 * H + dH + 2 * L * sqrt(H * (H + dH) / pow(L, 2) - pow(dH, 2))
        Cd = pow(L, 2) - pow(2 * H + dH, 2)

        dCndH = 2 + L * (2 * H + dH) / sqrt(H * (H + dH) * pow(L, 2) - pow(dH, 2))
        dCddH = -4 * (2 * H + dH)
        dCnddH = 1 + (L * H * (pow(L, 2) + 2 * H * dH + pow(dH, 2))) / (pow(pow(H, 2) - pow(dH, 2), 1.5) * sqrt(H * ( H + dH)))
        dCdddH = -2 * (2 * H + dH)

        dCdH = 2 * (dCndH * Cd - dCddH * Cn) / pow(Cd, 2)
        dCddH = 2 * (dCnddH * Cd - dCdddH * Cn) / pow(Cd, 2)

        u = (C + LdD * dCdH) / (C * sinh( C * (D + dD)))
        p = (C + LD * dCdH) / (C * sinh( C * D))
        q = (LD * dCdH) / (C * sinh(C * D))
        v = (C + LdD * dCddH) / (C * sinh( C * (D + dD)))

        M = zeros( (3, 3) )
        M[0, 0] = sqrt(1 - pow( b, 2)) / ((u + p) * self.maximum_H)
        M[0, 1] = b / ((u + p) * self.maximum_H)
        M[0, 2] = (v + q) / ((u + p) * self.maximum_H)

        M[1, 0] = -b * sqrt(1 - pow( b, 2)) / ( 2 * D + dD )
        M[1, 1] = (1 - pow( b, 2)) / ( 2 * D + dD )

        M[2, 2] = 1 / ( 2 * self.maximum_dH )

        interaction_matrix = M @ T

        self.raw_result = -self.gain * pinv( interaction_matrix ) @ (feature - self.target_feature)

        self.result = self.actuation_projection_matrix @ self.raw_result
        
        if self.record:
            self.compute_times.append( perf_counter() - ti )

        if self.verbose:
            print(f'{C=}\t{H=}\t{dH=}\t{D=}\t{dD=}')
            print(f'{feature=}')
            print(f'{angle=}')
            print(f'{M=}')
            print(f'{T=}')
            print(f'{interaction_matrix=}')
            print(f'{self.raw_result=}')
            print(f'{self.result=}')
            if self.record: print(f'{self.compute_times[-1]=}')

        return self.result
