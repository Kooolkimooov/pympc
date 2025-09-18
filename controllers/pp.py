from time import perf_counter

from numpy import eye, inf, ndarray, pow, zeros
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
    replan_method: str
        method to use for replanning the trajectory, must be one of the following:
        'never': never replan the trajectory
        'on_max_error': replan the trajectory if the error between the current state and the target trajectory
        exceeds max_replan_error
        'after_n_steps': replan the trajectory after steps_before_replan steps
        'error_and_steps': replan the trajectory if the error between the current state and the target trajectory
        exceeds max_replan_error or after steps_before_replan steps
        'always': replan the trajectory at each step
    max_replan_error: float
        maximum error between the current state and the target trajectory to replan the trajectory
    steps_before_replan: int
        number of steps before replanning the trajectory
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
    **get_trajectory**( *ndarray* ):
        computes an actionable trajectory over the horizon from the candidate
    **compute_result**():
        computes the best actuation from raw result and store it in self.result
    **should_replan**():
        checks if the trajectory should be replanned

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
    **max_replan_error**: **float**:
        maximum error between the current state and the target trajectory to replan the trajectory
    **steps_before_replan**: **int**:
        number of steps before replanning the trajectory
    **steps_since_replan**: **int**:
        number of steps since the last replanning
    **result**: *ndarray*:
        best actuation found during the optimization
    **raw_result**: *OptimizeResult*:
        raw result of the optimization
    **result_shape**: *tuple*:
        shape of the result array; (horizon, 1, actuation_size)
    """

    REPLAN_METHOD = [ 'never', 'at_end_of_horizon', 'on_max_error', 'after_n_steps', 'on_error_and_steps', 'always' ]

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
            replan_method: str = 'never',
            max_replan_error: float = 1.0,
            steps_before_replan: int = 1,
            record: bool = False,
            verbose: bool = False
    ):

        if replan_method == 'never':
            self.should_replan = self._should_replan_never
        if replan_method == 'at_end_of_horizon':
            self.should_replan = self._should_replan_at_end_of_horizon
        elif replan_method == 'on_max_error':
            self.should_replan = self._should_replan_on_max_error
        elif replan_method == 'after_n_steps':
            self.should_replan = self._should_replan_after_n_steps
        elif replan_method == 'on_error_and_steps':
            self.should_replan = self._should_replan_on_error_and_steps
        elif replan_method == 'always':
            self.should_replan = self._should_replan_always
        else:
            raise ValueError( f'replan_method must be one of {self.REPLAN_METHOD}' )

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

        self.raw_result = OptimizeResult( x=zeros( self.result_shape, ), success=False )
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

        self.max_replan_error = max_replan_error
        self.steps_before_replan = steps_before_replan
        self.steps_since_replan = 0
        self.is_first_plan = True

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
            ti = perf_counter()

        if not self.raw_result.success or self.should_replan():
            if self.verbose: print( 'planning' )

            self.predicted_trajectories.clear()
            self.steps_since_replan = 0

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
            if not self.raw_result.success and self.best_cost < inf:
                self.raw_result.x = self.best_candidate
                self.best_cost = inf
        else:
            if self.verbose: print( 'using previously planned trajectory' )
            temp = zeros( self.result_shape )
            temp[ :-1 ] = self.raw_result.x.reshape( self.result_shape )[ 1: ]
            temp[ -1 ] = temp[ -2 ]
            self.raw_result.x = temp

        if self.record:
            self.compute_times.append( perf_counter() - ti )

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

        trajectory = self.get_trajectory( candidate )

        error = self.model.dynamics.compute_error(
                trajectory, self.target_trajectory[ :self.horizon ]
        )

        cost = 0.

        cost += (error @ self.pose_weight_matrix @ error.transpose( (0, 2, 1) )).sum()
        cost += 0. if self.objective is None else self.objective_weight * self.objective(
                candidate
        )

        cost /= self.horizon

        cost += self.final_weight * (error[ -1 ] @ self.pose_weight_matrix[ -1 ] @ error[ -1 ].T).sum()

        if self.record:
            self.predicted_trajectories.append( trajectory )

        if cost < self.best_cost:
            self.best_candidate = candidate.copy()
            self.best_cost = cost

        return cost

    def get_trajectory( self, candidate: ndarray ) -> ndarray:
        """
        computes an actionable actuation over the horizon from the candidate actuation

        Parameters
        ----------
        candidate : ndarray
            proposed solution for the optimization problem; shape = (horizon, 1, actuation_size)

        Returns
        -------
        actuation: ndarray
            proposed actuation over the horizon; shape = (horizon, 1, actuation_size)
        actuation_derivatives: ndarray
            proposed actuation derivative over the horizon; shape = (horizon, 1, actuation_size)
        """
        trajectory_derivatives_body = candidate.reshape( self.result_shape )
        transform = self.model.dynamics.get_body_to_world_transform( self.model.state ).T
        trajectory_derivatives_world = trajectory_derivatives_body @ transform
        trajectory = trajectory_derivatives_world.cumsum( axis=0 ) * self.time_step + self.model.state[
            :self.model.dynamics.state_size // 2 ]

        return trajectory

    def compute_result( self ):
        """
        computes the best actuation from scipy.optimize raw result and store it in self.result
        """
        self.result = self.get_trajectory( self.raw_result.x )[ 0, 0, : ]

    def get_objective( self ) -> float:
        return self.objective( self.raw_result.x.reshape( self.result_shape ) )

    def should_replan( self ) -> bool:
        raise NotImplementedError

    def _should_replan_never( self ) -> bool:
        self.steps_since_replan += 1

        if self.is_first_plan:
            self.is_first_plan = False
            return True

        return False

    def _should_replan_at_end_of_horizon( self ) -> bool:
        self.steps_since_replan += 1

        if self.is_first_plan or self.steps_since_replan >= self.horizon:
            self.is_first_plan = False
            return True

        return False

    def _should_replan_on_max_error( self ) -> bool:
        self.steps_since_replan += 1

        error = self.model.dynamics.compute_error(
                self.model.state[ :self.model.dynamics.state_size // 2 ].reshape(
                        (1, 1, self.model.dynamics.state_size // 2)
                ),
                self.result.reshape( (1, 1, self.model.dynamics.state_size // 2) )
        )

        return pow( error, 2 ).sum() > pow( self.max_replan_error, 2 )

    def _should_replan_after_n_steps( self ) -> bool:
        self.steps_since_replan += 1

        if self.steps_since_replan >= self.steps_before_replan:
            return True

        return False

    def _should_replan_on_error_and_steps( self ) -> bool:
        self.steps_since_replan -= 1 # compensate double count
        return self._should_replan_on_max_error() or self._should_replan_after_n_steps()

    def _should_replan_always( self ) -> bool:
        return True
