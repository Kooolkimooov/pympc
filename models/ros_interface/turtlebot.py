from geometry_msgs.msg import Pose, Twist
from numpy import array, ndarray, pi, zeros
from scipy.spatial.transform import Rotation

from pympc.models.dynamics.turtlebot import Turtlebot
from pympc.models.ros_interface.interface import Interface


class TurtlebotROSInterface( Interface ):

  _command_type = Twist
  _initial_state = zeros( (Turtlebot._state_size,) )

  max_linear_speed = 0.65
  max_angular_speed = pi

  @staticmethod
  def ros_pose_from_state( state: ndarray ) -> Pose:
    pose = Pose()
    pose.position.x = state[ Turtlebot._position ][ 0 ]
    pose.position.y = state[ Turtlebot._position ][ 1 ]

    quaternion = Rotation.from_euler( 'xyz', array( [ 0, 0, state[ Turtlebot._orientation ] ] ) ).as_quat()
    pose.orientation.x = quaternion[ 0 ]
    pose.orientation.y = quaternion[ 1 ]
    pose.orientation.z = quaternion[ 2 ]
    pose.orientation.w = quaternion[ 3 ]

    return pose

  @staticmethod
  def pose_from_ros_pose( ros_pose: Pose ) -> ndarray:
    pose = zeros( (Turtlebot._state_size // 2,) )
    pose[ Turtlebot._position ][ 0 ] = ros_pose.position.x
    pose[ Turtlebot._position ][ 1 ] = ros_pose.position.y
    pose[ Turtlebot._orientation ] = Rotation.from_quat(
        [ ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w ]
        ).as_euler( 'xyz' )[ 2 ]
    return pose

  @staticmethod
  def actuation_from_ros_actuation( ros_actuation: any ) -> ndarray:
    actuation = zeros( (Turtlebot._actuation_size,) )

    actuation[ Turtlebot._linear_actuation ] = ros_actuation.linear.x
    actuation[ Turtlebot._angular_actuation ] = ros_actuation.angular.z

    if actuation[ Turtlebot._linear_actuation ] > TurtlebotROSInterface.max_linear_speed:
      actuation[ Turtlebot._linear_actuation ] = TurtlebotROSInterface.max_linear_speed
    elif actuation[ Turtlebot._linear_actuation ] < -TurtlebotROSInterface.max_linear_speed:
      actuation[ Turtlebot._linear_actuation ] = -TurtlebotROSInterface.max_linear_speed
    if actuation[ Turtlebot._angular_actuation ] > TurtlebotROSInterface.max_angular_speed:
      actuation[ Turtlebot._angular_actuation ] = TurtlebotROSInterface.max_angular_speed
    elif actuation[ Turtlebot._angular_actuation ] < -TurtlebotROSInterface.max_angular_speed:
      actuation[ Turtlebot._angular_actuation ] = -TurtlebotROSInterface.max_angular_speed

    return actuation

  @staticmethod
  def ros_actuation_from_actuation( actuation: ndarray ) -> any:
    ros_actuation = TurtlebotROSInterface._command_type()

    ros_actuation.linear.x = actuation[ Turtlebot._linear_actuation ]
    ros_actuation.angular.z = actuation[ Turtlebot._angular_actuation ]

    return ros_actuation
