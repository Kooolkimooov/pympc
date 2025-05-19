from geometry_msgs.msg import Pose
from mavros_msgs.msg import OverrideRCIn
from numpy import array, ndarray, zeros
from scipy.spatial.transform import Rotation

from pympc.models.dynamics.bluerov import Bluerov
from pympc.models.ros_interface.interface import Interface
from pympc.utils import G


class BluerovROSInterface( Interface ):

  _command_type = OverrideRCIn
  _initial_state = zeros( (Bluerov._state_size,) )

  # TODO: refine values
  max_kg_force = 3
  n_thrusters = 4
  cos_angle = 0.7071067812
  neutral_pwm = 1500
  half_range = 400
  distance_from_com = 0.1

  @staticmethod
  def get_max_actuation() -> ndarray:
    max_actuation = array(
        [ BluerovROSInterface.n_thrusters * BluerovROSInterface.max_kg_force * G * BluerovROSInterface.cos_angle,
          BluerovROSInterface.n_thrusters * BluerovROSInterface.max_kg_force * G * BluerovROSInterface.cos_angle,
          BluerovROSInterface.n_thrusters * BluerovROSInterface.max_kg_force * G,
          BluerovROSInterface.n_thrusters * BluerovROSInterface.max_kg_force * G *
          BluerovROSInterface.distance_from_com,
          BluerovROSInterface.n_thrusters * BluerovROSInterface.max_kg_force * G *
          BluerovROSInterface.distance_from_com,
          BluerovROSInterface.n_thrusters * BluerovROSInterface.max_kg_force * G *
          BluerovROSInterface.distance_from_com ]
        )
    return max_actuation

  @staticmethod
  def ros_pose_from_state( state: ndarray ) -> Pose:
    pose = Pose()
    pose.position.x = state[ Bluerov._position[ 0 ] ]
    pose.position.y = state[ Bluerov._position[ 1 ] ]
    pose.position.z = state[ Bluerov._position[ 2 ] ]

    quaternion = Rotation.from_euler( 'xyz', state[ Bluerov._orientation ] ).as_quat()
    pose.orientation.x = quaternion[ 0 ]
    pose.orientation.y = quaternion[ 1 ]
    pose.orientation.z = quaternion[ 2 ]
    pose.orientation.w = quaternion[ 3 ]

    return pose

  @staticmethod
  def pose_from_ros_pose( ros_pose: Pose ) -> ndarray:
    pose = zeros( (Bluerov._state_size // 2,) )
    pose[ Bluerov._position[ 0 ] ] = ros_pose.position.x
    pose[ Bluerov._position[ 1 ] ] = ros_pose.position.y
    pose[ Bluerov._position[ 2 ] ] = ros_pose.position.z
    pose[ 3:6 ] = Rotation.from_quat(
        [ ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w ]
        ).as_euler( 'xyz' )

    return pose

  @staticmethod
  def pwm_to_normalized( value: int ) -> float:
    return (value - BluerovROSInterface.neutral_pwm) / BluerovROSInterface.half_range

  @staticmethod
  def normalized_to_pwm( value: float ) -> int:
    pwm = int( value * BluerovROSInterface.half_range ) + BluerovROSInterface.neutral_pwm
    return pwm

  @staticmethod
  def actuation_from_ros_actuation( ros_actuation: any ) -> ndarray:
    actuation = zeros( (Bluerov._actuation_size,) )

    force = BluerovROSInterface.n_thrusters * BluerovROSInterface.max_kg_force * G
    angle = force * BluerovROSInterface.cos_angle
    lever = force * BluerovROSInterface.distance_from_com

    # TODO this should use the bluerov configuration
    actuation[ 0 ] = angle * BluerovROSInterface.pwm_to_normalized( ros_actuation.channels[ 4 ] )
    actuation[ 1 ] = angle * BluerovROSInterface.pwm_to_normalized( ros_actuation.channels[ 5 ] )
    actuation[ 2 ] = force * BluerovROSInterface.pwm_to_normalized( ros_actuation.channels[ 2 ] )
    actuation[ 3 ] = lever * BluerovROSInterface.pwm_to_normalized( ros_actuation.channels[ 1 ] )
    actuation[ 4 ] = lever * BluerovROSInterface.pwm_to_normalized( ros_actuation.channels[ 0 ] )
    actuation[ 5 ] = lever * BluerovROSInterface.pwm_to_normalized( ros_actuation.channels[ 3 ] )

    return actuation

  @staticmethod
  def ros_actuation_from_actuation( actuation: ndarray ) -> any:
    ros_actuation = BluerovROSInterface._command_type()

    force = BluerovROSInterface.n_thrusters * BluerovROSInterface.max_kg_force * G
    angle = force * BluerovROSInterface.cos_angle
    lever = force * BluerovROSInterface.distance_from_com

    # TODO: clamp here
    # TODO this should use the bluerov configuration

    ros_actuation.channels[ 4 ] = BluerovROSInterface.normalized_to_pwm( actuation[ 0 ] / angle )
    ros_actuation.channels[ 5 ] = BluerovROSInterface.normalized_to_pwm( actuation[ 1 ] / angle )
    ros_actuation.channels[ 2 ] = BluerovROSInterface.normalized_to_pwm( actuation[ 2 ] / force )
    ros_actuation.channels[ 1 ] = BluerovROSInterface.normalized_to_pwm( actuation[ 3 ] / lever )
    ros_actuation.channels[ 0 ] = BluerovROSInterface.normalized_to_pwm( actuation[ 4 ] / lever )
    ros_actuation.channels[ 3 ] = BluerovROSInterface.normalized_to_pwm( actuation[ 5 ] / lever )

    return ros_actuation
