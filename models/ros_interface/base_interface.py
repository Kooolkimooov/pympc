from numpy import ndarray

from geometry_msgs.msg import Pose


class BaseInterface:
  command_type = None
  initial_state = None

  @staticmethod
  def ros_pose_from_state( state: ndarray ) -> any: raise NotImplementedError()

  @staticmethod
  def pose_from_ros_pose( ros_pose: Pose ) -> ndarray: raise NotImplementedError()

  @staticmethod
  def actuation_from_ros_actuation( ros_actuation: any ) -> ndarray: raise NotImplementedError()

  @staticmethod
  def ros_actuation_from_actuation( actuation: ndarray ) -> any: raise NotImplementedError()
