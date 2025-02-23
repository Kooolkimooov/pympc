from copy import deepcopy
from time import perf_counter

from numpy import eye, inf, ndarray, zeros
from scipy.optimize import minimize, OptimizeResult

class PP:

  def __init__(
      self,
      current_pose: ndarray,
      target: ndarray,
      horizon: int,
      objective: callable = None,
      time_step: float = .01,
      guess_from_last_solution: bool = True,
      tolerance: float = 1e-6,
      max_number_of_iteration: int = 1000,
      bounds: tuple = None,
      constraints: tuple = None,
      pose_weight_matrix: ndarray = None,
      objective_weight: float = 0.,
      final_weight: float = 0.,
      record: bool = False,
      verbose: bool = False
      ):
    """
    :param horizon: prediction horizon
    :param target: target
    :param objective: objective function, must have the following signature: f(path)
    :param time_step:time step
    :param guess_from_last_solution: whether to use the last solution as the initial guess
    :param tolerance: tolerance for the optimization algorithm
    :param max_number_of_iteration: maximum number of iterations for the optimization algorithm
    :param bounds: bounds for the optimization variables
    :param constraints: constraints for the optimization variables
    :param pose_weight_matrix: weight matrix for the pose error
    :param objective_weight: weight for the objective function
    :param final_weight: weight for the final pose error
    :param record: whether to record the computation times, predicted trajectories and candidate
    actuations
    :param verbose: whether to print the optimization results
    """

    self.current_pose = current_pose
    self.target = target.reshape((1, 1, target.shape[0])).repeat(horizon, axis=0)
    self.horizon = horizon
    self.objective = objective

    self.time_step = time_step

    self.guess_from_last_solution = guess_from_last_solution
    self.tolerance = tolerance
    self.max_number_of_iteration = max_number_of_iteration
    self.bounds = bounds
    self.constraints = constraints

    self.result_shape = ( self.horizon, 1, self.current_pose.shape[ 0 ] )

    self.raw_result = OptimizeResult( x = zeros( self.result_shape ) )
    self.result = zeros( self.result_shape )

    self.pose_weight_matrix: ndarray = zeros(
        (self.horizon, self.current_pose.shape[0], self.current_pose.shape[ 0 ] )
        )

    if pose_weight_matrix is None:
      self.pose_weight_matrix[ : ] = eye( self.current_pose.shape[ 0 ] )
    else:
      self.pose_weight_matrix[ : ] = pose_weight_matrix

    self.objective_weight = objective_weight
    self.final_weight = final_weight

    self.best_cost = inf
    self.best_candidate = zeros( self.result_shape )

    self.record = record
    if self.record:
      self.predicted_paths = [ ]
      self.compute_times = [ ]

    self.verbose = verbose

  def compute_path( self ) -> ndarray:
    """
    computes the best path for the current pose with a given horizon. records the computation
    time if record is True and returns the best actuation
    :return: best actuation
    """

    if self.record:
      self.predicted_paths.clear()
      ti = perf_counter()

    self.raw_result = minimize(
        fun = self.cost,
        x0 = self.raw_result.x.flatten(),
        tol = self.tolerance,
        bounds = self.bounds,
        constraints = self.constraints,
        options = {
            'maxiter': self.max_number_of_iteration, 'disp': self.verbose
            }
        )

    if self.record:
      self.compute_times.append( perf_counter() - ti )

    if self.raw_result.success:
      self.get_result()
    elif self.best_cost < inf:
      self.raw_result.x = self.best_candidate
      self.get_result()

    return self.result

  def get_result( self ):
    """
    computes the best actuation from scipy.optimize raw result and store it in self.result
    """
    self.result = self.raw_result.x.copy()

  def cost( self, candidate: ndarray ) -> float:
    """
    cost function for the optimization. records the predicted trajectories and candidate actuation
    :param candidate: proposed actuation derivative over the horizon
    :return: cost
    """
    
    cost = 0.

    path = candidate.reshape( self.result_shape )

    error = path - self.target
    cost += (error @ self.pose_weight_matrix @ error.transpose( (0, 2, 1) )).sum()
    cost += 0. if self.objective is None else self.objective_weight * self.objective( path )

    cost /= self.horizon

    cost += self.final_weight * (error[ -1 ] @ self.pose_weight_matrix[ -1 ] @ error[ -1 ].T).sum()

    if self.record:
      self.predicted_paths.append( path )

    if cost < self.best_cost:
      self.best_candidate = candidate.copy()

    return cost

  def get_objective( self ) -> float:
    return self.objective( self.result )


if __name__ == "__main__":
  from numpy import ones, diff
  from scipy.optimize import NonlinearConstraint

  def constraint(self: PP, path: ndarray) -> ndarray:
    nf_path = path.reshape(self.result_shape)
    return diff(nf_path, prepend=[[self.current_pose]], axis=0).flatten()
  
  ub = 0.05 * ones((10, 1, 6))
  lb = -ub

  o = PP(
      current_pose = zeros( 6 ),
      target = ones( 6 ),
      horizon = 10,
      record=True
      )
  
  o.constraints_function = constraint.__get__( o, PP )
  o.constraints = NonlinearConstraint(o.constraints_function, lb.flatten(), ub.flatten())

  o.compute_path()
  print(o.result)
  print(o.compute_times)