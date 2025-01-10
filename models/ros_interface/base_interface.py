from numpy import ndarray


class BaseInterface:
  command_type = None
  initial_state = None

  @staticmethod
  def ros_pose_from_state( state: ndarray ) -> any: raise NotImplementedError()

  @staticmethod
  def actuation_from_ros_actuation( ros_actuation: any ) -> ndarray: raise NotImplementedError()
