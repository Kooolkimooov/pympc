from numpy import array, ndarray, pi, zeros
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Pose, Twist
from .base_interface import BaseInterface
from ..dynamics.turtlebot import Turtlebot


class TurtlebotROSInterface( BaseInterface ):

  command_type = Twist
  initial_state = zeros( (6,) )

  max_linear_speed = 0.65
  max_angular_speed = pi

  @staticmethod
  def ros_pose_from_state( state: ndarray ) -> Pose:
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
  def pose_from_ros_pose( ros_pose: Pose ) -> ndarray:
    pose = zeros( (Turtlebot.pose_size,) )
    pose[ 0 ] = ros_pose.position.x
    pose[ 1 ] = ros_pose.position.y
    pose[ 2 ] = Rotation.from_quat(
        [ ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w ]
        ).as_euler( 'xyz' )[2]
    return pose

  @staticmethod
  def actuation_from_ros_actuation( ros_actuation: any ) -> ndarray:
    actuation = zeros( (Turtlebot.actuation_size,) )

    actuation[ 0 ] = ros_actuation.linear.x
    actuation[ 1 ] = ros_actuation.angular.z

    if actuation[ 0 ] > TurtlebotROSInterface.max_linear_speed:
      actuation[ 0 ] = TurtlebotROSInterface.max_linear_speed
    elif actuation[ 0 ] < -TurtlebotROSInterface.max_linear_speed:
      actuation[ 0 ] = -TurtlebotROSInterface.max_linear_speed
    if actuation[ 1 ] > TurtlebotROSInterface.max_angular_speed:
      actuation[ 1 ] = TurtlebotROSInterface.max_angular_speed
    elif actuation[ 1 ] < -TurtlebotROSInterface.max_angular_speed:
      actuation[ 1 ] = -TurtlebotROSInterface.max_angular_speed

    return actuation

  @staticmethod
  def ros_actuation_from_actuation( actuation: ndarray ) -> any:
    ros_actuation = TurtlebotROSInterface.command_type()

    ros_actuation.linear.x = actuation[ 0 ]
    ros_actuation.angular.z = actuation[ 1 ]

    return ros_actuation
