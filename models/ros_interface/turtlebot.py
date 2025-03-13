from numpy import array, ndarray, pi, zeros
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Pose, Twist
from .interface import Interface
from ..dynamics.turtlebot import Turtlebot

# instanciation to be able to use the properties
Turtlebot = Turtlebot()


class TurtlebotROSInterface( Interface ):

  _command_type = Twist
  _initial_state = zeros( (6,) )

  max_linear_speed = 0.65
  max_angular_speed = pi

  @staticmethod
  def ros_pose_from_state( state: ndarray ) -> Pose:
    pose = Pose()
    pose.position.x = state[ Turtlebot.position ][ 0 ]
    pose.position.y = state[ Turtlebot.position ][ 1 ]

    quaternion = Rotation.from_euler( 'xyz', array( [ 0, 0, state[ Turtlebot.orientation ] ] ) ).as_quat()
    pose.orientation.x = quaternion[ 0 ]
    pose.orientation.y = quaternion[ 1 ]
    pose.orientation.z = quaternion[ 2 ]
    pose.orientation.w = quaternion[ 3 ]

    return pose

  @staticmethod
  def pose_from_ros_pose( ros_pose: Pose ) -> ndarray:
    pose = zeros( (Turtlebot.state_size // 2,) )
    pose[ Turtlebot.position ][ 0 ] = ros_pose.position.x
    pose[ Turtlebot.position ][ 1 ] = ros_pose.position.y
    pose[ Turtlebot.orientation ] = Rotation.from_quat(
        [ ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w ]
        ).as_euler( 'xyz' )[ 2 ]
    return pose

  @staticmethod
  def actuation_from_ros_actuation( ros_actuation: any ) -> ndarray:
    actuation = zeros( (Turtlebot.actuation_size,) )

    actuation[ Turtlebot.linear_actuation ] = ros_actuation.linear.x
    actuation[ Turtlebot.angular_actuation ] = ros_actuation.angular.z

    if actuation[ Turtlebot.linear_actuation ] > TurtlebotROSInterface.max_linear_speed:
      actuation[ Turtlebot.linear_actuation ] = TurtlebotROSInterface.max_linear_speed
    elif actuation[ Turtlebot.linear_actuation ] < -TurtlebotROSInterface.max_linear_speed:
      actuation[ Turtlebot.linear_actuation ] = -TurtlebotROSInterface.max_linear_speed
    if actuation[ Turtlebot.angular_actuation ] > TurtlebotROSInterface.max_angular_speed:
      actuation[ Turtlebot.angular_actuation ] = TurtlebotROSInterface.max_angular_speed
    elif actuation[ Turtlebot.angular_actuation ] < -TurtlebotROSInterface.max_angular_speed:
      actuation[ Turtlebot.angular_actuation ] = -TurtlebotROSInterface.max_angular_speed

    return actuation

  @staticmethod
  def ros_actuation_from_actuation( actuation: ndarray ) -> any:
    ros_actuation = TurtlebotROSInterface.command_type()

    ros_actuation.linear.x = actuation[ Turtlebot.linear_actuation ]
    ros_actuation.angular.z = actuation[ Turtlebot.angular_actuation ]

    return ros_actuation
