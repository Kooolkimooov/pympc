from geometry_msgs.msg import Pose
from numpy import ndarray


class Interface:
  _command_type = None
  _initial_state = None

  @staticmethod
  def ros_pose_from_state( state: ndarray ) -> Pose:
    """
    converts a numpy array to a ROS Pose message

    :param state: numpy array represeting the state
    :returns: Pose message
    """
    raise NotImplementedError()

  @staticmethod
  def pose_from_ros_pose( ros_pose: Pose ) -> ndarray:
    """
    converts a ROS Pose message to a numpy array
    
    :param ros_pose: Pose message
    :returns: numpy array represeting the state
    """
    raise NotImplementedError()

  @staticmethod
  def actuation_from_ros_actuation( ros_actuation: any ) -> ndarray:
    """
    converts a ROS message of type command_type to a numpy array

    :param ros_actuation: command_type message 
    :returns: numpy array represeting the actuation
    """
    raise NotImplementedError()

  @staticmethod
  def ros_actuation_from_actuation( actuation: ndarray ) -> any:
    """
    converts a numpy array to a ROS message of type command_type

    :param actuation: numpy array represeting the actuation
    :returns: command_type message
    """
    raise NotImplementedError()

  @property
  def command_type( self ) -> any:
    """
    The type of the ros message used to send commands to the robot
    """
    return self._command_type

  @property
  def initial_state( self ) -> ndarray:
    """
    the initial state of the robot
    """
    return self._initial_state
