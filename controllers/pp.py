from time import perf_counter

from numpy import eye, inf, ndarray, zeros
from scipy.optimize import OptimizeResult, minimize

from pympc.models.model import Model


class PP:
    """
    implements a model path planning for a given model.

    Parameters
    ----------
    horizon: int
        prediction horizon
    target_trajectory: ndarray
        target trajectory
    model: Model
        model of the system
    objective: callable
        objective function, must have the following signature: f(trajectory)
    guess_from_last_solution: bool
        whether to use the last solution as the initial guess
    tolerance: float
        tolerance for the optimization algorithm
    max_number_of_iteration: int
        maximum number of iterations for the optimization algorithm
    bounds: Bounds | tuble(Bounds)
        bounds for the optimization variables
    constraints: LinearConstraints | NonLinearConstraints | tuple(LinearConstraints) | tuple(NonLinearConstraints)
        constraints for the optimization variables
    pose_weight_matrix: ndarray
        weight matrix for the pose error; shape: (state_dim//2, state_dim//2)
    objective_weight: float
        weight for the objective function
    final_weight: float
        weight for the final pose error
    record: bool
        whether to record the computation times, predicted trajectories and candidate actuations
    verbose: bool
        whether to print the optimization results

    Methods
    -------
    **step**():
        computes the best pose for the current state of the model
    **cost**( *ndarray* ):
        cost function for the optimization
    **get_objective**():
        computes the objective function value for the current state of the model

    Properties
    ----------
    **horizon**: *int*:
        prediction horizon
    **target_trajectory**: *ndarray*:
        target trajectory
    **objective**: *callable*:
        objective function, must have the following signature: f(trajectory, actuation)
    **time_step**: *float*:
        time step of the MPC prediction, defaults to the model time step
    **guess_from_last_solution**: *bool*:
        whether to use the last solution as the initial guess
    **tolerance**: *float*:
        tolerance for the optimization algorithm
    **max_number_of_iteration**: *int*:
        maximum number of iterations for the optimization algorithm
    **bounds**: *Bounds* | tuple(Bounds):
        bounds for the optimization variables
    **constraints**: *LinearConstraints* | NonLinearConstraints | tuple(LinearConstraints) | tuple(
    NonLinearConstraints):
        constraints for the optimization problem
    **pose_weight_matrix**: *ndarray*:
        weight matrix for the pose error; shape: (horizon, state_size//2, state_size//2)
    **objective_weight**: *float*:
        weight for the objective function
    **final_weight**: *float*:
        weight for the final pose error
    **record**: *bool*:
        whether to record the computation times, predicted trajectories and candidate actuations
    **verbose**: *bool*:
        whether to print the optimization results
    **predicted_trajectories**: *list*:
        list of predicted trajectories; reset before each optimization
    **compute_times**: *list*:
        list of computation times
    **best_cost**: *float*:
        best cost found during the optimization
    **best_candidate**: *ndarray*:
        best candidate found during the optimization; returned if optimization fails
    **result**: *ndarray*:
        best actuation found during the optimization
    **raw_result**: *OptimizeResult*:
        raw result of the optimization
    **result_shape**: *tuple*:
        shape of the result array; (horizon, 1, actuation_size)
    """

    def __init__(
            self,
            horizon: int,
            target_trajectory: ndarray,
            model: Model,
            objective: callable = None,
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
        self.model = model

        self.horizon = horizon
        self.target_trajectory = target_trajectory
        self.objective = objective

        self.time_step = self.model.time_step

        self.guess_from_last_solution = guess_from_last_solution
        self.tolerance = tolerance
        self.max_number_of_iteration = max_number_of_iteration
        self.bounds = bounds
        self.constraints = constraints

        self.result_shape = (
                self.horizon, 1, self.model.dynamics.state_size // 2
        )

        self.raw_result = OptimizeResult( x=zeros( self.result_shape, ) )
        self.result = zeros( (self.model.dynamics.state_size // 2) )

        self.pose_weight_matrix: ndarray = zeros(
                (self.horizon, self.model.dynamics.state_size // 2, self.model.dynamics.state_size // 2)
        )

        if pose_weight_matrix is None:
            self.pose_weight_matrix[ : ] = eye( self.model.state.shape[ 0 ] // 2 )
        else:
            self.pose_weight_matrix[ : ] = pose_weight_matrix

        self.objective_weight = objective_weight
        self.final_weight = final_weight

        self.best_cost = inf
        self.best_candidate = zeros( self.result_shape )

        self.record = record

        self.predicted_trajectories = [ ]
        self.compute_times = [ ]

        self.verbose = verbose

    def step( self ) -> ndarray:
        """
        computes the best pose for the current state with a given horizon. records the computation
        time if record is True and returns the best actuation

        Returns
        -------
        ndarray:
            best pose for the next step; shape = (actuation_size,)
        """

        if self.record:
            self.predicted_trajectories.clear()
            ti = perf_counter()

        self.raw_result = minimize(
                fun=self.cost,
                x0=self.raw_result.x.flatten(),
                tol=self.tolerance,
                bounds=self.bounds,
                constraints=self.constraints,
                options={
                        'maxiter': self.max_number_of_iteration, 'disp': self.verbose
                }
        )

        if self.record:
            self.compute_times.append( perf_counter() - ti )

        if self.raw_result.success:
            self.compute_result()
        elif self.best_cost < inf:
            self.raw_result.x = self.best_candidate
            self.best_cost = inf
            self.compute_result()

        return self.result

    def cost( self, candidate: ndarray ) -> float:
        """
        cost function for the optimization. records the predicted trajectories and candidate actuation
        :param candidate: proposed actuation derivative over the horizon

        Parameters
        ----------
        candidate: ndarray
            proposed solution for the optimization problem; shape = (horizon, 1, actuation_size)

        Returns
        -------
        float:
            cost of the candidate for the optimization
        """

        candidate = candidate.reshape( self.result_shape )

        error = self.model.dynamics.compute_error(
                candidate, self.target_trajectory[ :self.horizon ]
        )

        cost = 0.

        cost += (error @ self.pose_weight_matrix @ error.transpose( (0, 2, 1) )).sum()
        cost += 0. if self.objective is None else self.objective_weight * self.objective(
                candidate
        )

        cost /= self.horizon

        cost += self.final_weight * (error[ -1 ] @ self.pose_weight_matrix[ -1 ] @ error[ -1 ].T).sum()

        if self.record:
            self.predicted_trajectories.append( candidate )

        if cost < self.best_cost:
            self.best_candidate = candidate.copy()

        return cost

    def compute_result( self ):
        """
        computes the best actuation from scipy.optimize raw result and store it in self.result
        """
        self.result = self.raw_result.x.reshape( self.result_shape )[ 0, 0, : ]

    def get_objective( self ) -> float:
        return self.objective( self.raw_result.x.reshape( self.result_shape ) )
