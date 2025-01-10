from numpy import ndarray, zeros
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Pose
from mavros_msgs.msg import OverrideRCIn
from pympc.models.dynamics.bluerov import Bluerov
from pympc.models.ros_interface.base_interface import BaseInterface
from pympc.utils import G


class BluerovROSInterface( BaseInterface ):

  command_type = OverrideRCIn
  initial_state = zeros( (Bluerov.state_size,) )

  @staticmethod
  def ros_pose_from_state( state: ndarray ) -> Pose:
    pose = Pose()
    pose.position.x = state[ 0 ]
    pose.position.y = state[ 1 ]
    pose.position.z = state[ 2 ]

    quaternion = Rotation.from_euler( 'xyz', state[ 3:6 ] ).as_quat()
    pose.orientation.x = quaternion[ 0 ]
    pose.orientation.y = quaternion[ 1 ]
    pose.orientation.z = quaternion[ 2 ]
    pose.orientation.w = quaternion[ 3 ]

    return pose

  @staticmethod
  def actuation_from_ros_actuation( ros_actuation ) -> ndarray:
    actuation = zeros( (Bluerov.actuation_size,) )

    # TODO: refine values
    max_kg_force = 3
    n_thrusters = 4
    cos_angle = 0.7071067812
    neutral_pwm = 1500
    half_range = 400
    distance_from_com = 0.1

    actuation[ 0 ] = n_thrusters * max_kg_force * G * cos_angle * (
        ros_actuation.channels[ 4 ] - neutral_pwm) / half_range
    actuation[ 1 ] = n_thrusters * max_kg_force * G * cos_angle * (
        ros_actuation.channels[ 5 ] - neutral_pwm) / half_range
    actuation[ 2 ] = n_thrusters * max_kg_force * G * (ros_actuation.channels[ 2 ] - neutral_pwm) / half_range
    actuation[ 3 ] = n_thrusters * max_kg_force * G * distance_from_com * (
        ros_actuation.channels[ 1 ] - neutral_pwm) / half_range
    actuation[ 4 ] = n_thrusters * max_kg_force * G * distance_from_com * (
        ros_actuation.channels[ 0 ] - neutral_pwm) / half_range
    actuation[ 5 ] = n_thrusters * max_kg_force * G * distance_from_com * (
        ros_actuation.channels[ 3 ] - neutral_pwm) / half_range

    return actuation
