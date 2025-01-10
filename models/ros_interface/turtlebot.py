from numpy import array, ndarray, pi, zeros
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Pose, Twist
from pympc.models.dynamics.turtlebot import Turtlebot
from pympc.models.ros_interface.base_interface import BaseInterface


class TurtlebotROSInterface( BaseInterface ):

  command_type = Twist
  initial_state = zeros( (3,) )

  @staticmethod
  def pose_from_state( state: ndarray ) -> Pose:
    pose = Pose()
    pose.position.x = state[ 0 ]
    pose.position.y = state[ 1 ]

    quaternion = Rotation.from_euler( 'xyz', array( [ 0, 0, state[ 2 ] ] ) ).as_quat()
    pose.orientation.x = quaternion[ 0 ]
    pose.orientation.y = quaternion[ 1 ]
    pose.orientation.z = quaternion[ 2 ]
    pose.orientation.w = quaternion[ 3 ]

    return pose

  @staticmethod
  def actuation_from_ros( actuation ) -> ndarray:
    actuation = zeros( (Turtlebot.actuation_size,) )

    max_linear_speed = 0.65
    max_angular_speed = pi

    actuation[ 0 ] = actuation.linear.x
    actuation[ 1 ] = actuation.angular.z

    if actuation[ 0 ] > max_linear_speed:
      actuation[ 0 ] = max_linear_speed
    elif actuation[ 0 ] < -max_linear_speed:
      actuation[ 0 ] = -max_linear_speed
    if actuation[ 1 ] > max_angular_speed:
      actuation[ 1 ] = max_angular_speed
    elif actuation[ 1 ] < -max_angular_speed:
      actuation[ 1 ] = -max_angular_speed

    return actuation
