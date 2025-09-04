from time import perf_counter

from numpy import diff, eye, inf, ndarray, zeros
from scipy.optimize import OptimizeResult, minimize

from pympc.models.model import Model


class MPC:
    """
    implements a model predictive controller for a given model. 

    PREDICTION ASSUMES THAT THE MODELS STATE IS X = [POSE, POSE_DERIVATIVE]
    
    Parameters
    ----------
    model: Model
        model of the system
    horizon: int
        prediction horizon
    target_trajectory: ndarray
        target trajectory
    objective: callable
        objective function, must have the following signature: f(trajectory, actuation)
    time_steps_per_actuation: int
        number of time steps per proposed actuation over the horizon
    guess_from_last_solution: bool
        whether to use the last solution as the initial guess
    use_cache_prediction: bool
        whether to cache the predictions to speed up the optimization, uses more memory
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
    actuation_weight_matrix: ndarray
        weight matrix for the actuation derivative; shape: (actuation_dim, actuation_dim)
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
    **compute_actuation**():
        computes the best actuation for the current state of the model
    **compute_result**():
        computes an actionable actuation from `raw_result` depending on the configuration
    **cost**( *ndarray* ):
        cost function for the optimization
    **predict**( *ndarray* ):
        predicts the trajectory given the candidate actuation over the horizon
    **get_actuation**( *ndarray* ):
        computes an actionable actuation over the horizon from the candidate actuation depending on the configuration
    **get_objective**():
        computes the objective function value for the current state of the model
    
    Properties
    ----------
    **model**: *Model*:
        model of the system
    **horizon**: *int*:
        prediction horizon
    **target_trajectory**: *ndarray*:
        target trajectory
    **objective**: *callable*:
        objective function, must have the following signature: f(trajectory, actuation)
    **time_step**: *float*:
        time step of the MPC prediction, defaults to the model time step
    **time_steps_per_actuation**: *int*:
        number of time steps per proposed actuation over the horizon
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
    **actuation_weight_matrix**: *ndarray*:
        weight matrix for the actuation derivative; shape: (horizon, actuation_size, actuation_size)
    **objective_weight**: *float*:
        weight for the objective function
    **final_weight**: *float*:
        weight for the final pose error
    **use_prediction_cache**: *bool*:
        whether to cache the predictions to speed up the optimization, uses more memory
    **prediction_cache**: *dict*:
        cache for the predicted trajectories
    **record**: *bool*:
        whether to record the computation times, predicted trajectories and candidate actuations
    **verbose**: *bool*:
        whether to print the optimization results
    **predicted_trajectories**: *list*:
        list of predicted trajectories; reset before each optimization
    **candidate_actuations**: *list*:
        list of candidate actuations; reset before each optimization
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

    MODEL_TYPE = [ 'linear', 'nonlinear' ]
    OPTIMIZE_ON = [ 'actuation_derivative', 'actuation' ]

    def __init__(
            self,
            model: Model,
            horizon: int,
            target_trajectory: ndarray,
            model_type: str = 'nonlinear',
            optimize_on: str = 'actuation_derivative',
            objective: callable = None,
            time_steps_per_actuation: int = 1,
            guess_from_last_solution: bool = True,
            use_cache_prediction: bool = True,
            tolerance: float = 1e-6,
            max_number_of_iteration: int = 1000,
            bounds: tuple = None,
            constraints: tuple = None,
            pose_weight_matrix: ndarray = None,
            actuation_weight_matrix: ndarray = None,
            objective_weight: float = 0.,
            final_weight: float = 0.,
            record: bool = False,
            verbose: bool = False
    ):
        if model_type == 'linear':
            self.predict = self._predict_linear
        elif model_type == 'nonlinear':
            self.predict = self._predict_non_linear
        else:
            raise ValueError( f'model_type must be one of {self.MODEL_TYPE}' )

        if optimize_on == 'actuation_derivative':
            self.get_actuation = self._get_actuation_from_derivative
            self.compute_result = self._compute_result_from_derivative
        elif optimize_on == 'actuation':
            self.get_actuation = self._get_actuation_from_actual
            self.compute_result = self._compute_result_from_actual
        else:
            raise ValueError( f'optimize_on must be one of {self.OPTIMIZE_ON}' )

        self.model = model
        self.horizon = horizon
        self.target_trajectory = target_trajectory
        self.objective = objective

        self.time_step = self.model.time_step

        self.time_steps_per_actuation = time_steps_per_actuation
        self.guess_from_last_solution = guess_from_last_solution
        self.tolerance = tolerance
        self.max_number_of_iteration = max_number_of_iteration
        self.bounds = bounds
        self.constraints = constraints

        add_one = (1 if self.horizon % self.time_steps_per_actuation != 0 else 0)
        self.result_shape = (
                self.horizon // self.time_steps_per_actuation + add_one, 1, self.model.actuation.shape[ 0 ]
        )

        self.raw_result = OptimizeResult( x=zeros( self.result_shape, ) )
        self.result = zeros( self.model.actuation.shape )

        self.pose_weight_matrix: ndarray = zeros(
                (self.horizon, self.model.state.shape[ 0 ] // 2, self.model.state.shape[ 0 ] // 2)
        )
        self.actuation_weight_matrix: ndarray = zeros(
                (self.horizon, self.result_shape[ 2 ], self.result_shape[ 2 ])
        )

        if pose_weight_matrix is None:
            self.pose_weight_matrix[ : ] = eye( self.model.state.shape[ 0 ] // 2 )
        else:
            self.pose_weight_matrix[ : ] = pose_weight_matrix

        if actuation_weight_matrix is None:
            self.actuation_weight_matrix[ : ] = eye( self.result_shape[ 2 ] )
        else:
            self.actuation_weight_matrix[ : ] = actuation_weight_matrix

        self.objective_weight = objective_weight
        self.final_weight = final_weight

        self.best_cost = inf
        self.best_candidate = zeros( self.result_shape )

        self.record = record

        self.predicted_trajectories = [ ]
        self.candidate_actuations = [ ]
        self.compute_times = [ ]

        self.verbose = verbose

        self.use_prediction_cache = use_cache_prediction

        self.prediction_cache = { }

    def compute_actuation( self ) -> ndarray:
        """
        computes the best actuation for the current state with a given horizon. records the computation
        time if record is True and returns the best actuation

        Returns
        -------
        ndarray:
            best actuation for the next step; shape = (actuation_size,)
        """

        if self.record:
            self.predicted_trajectories.clear()
            self.candidate_actuations.clear()
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

    def compute_result( self ):
        """
        computes the best actuation from scipy.optimize raw result and store it in self.result 
        """
        raise NotImplementedError( 'predict method should have been implemented in __init__' )

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

        actuation, actuation_derivatives = self.get_actuation( candidate )

        prediction = self.predict( actuation )
        predicted_trajectory = prediction[ :, :, :self.model.state.shape[ 0 ] // 2 ]

        error = self.model.dynamics.compute_error(
                predicted_trajectory, self.target_trajectory[ :self.horizon ]
        )

        cost = 0.

        cost += (error @ self.pose_weight_matrix @ error.transpose( (0, 2, 1) )).sum()
        cost += (actuation_derivatives @ self.actuation_weight_matrix @ actuation_derivatives.transpose(
                (0, 2, 1)
        )).sum()
        cost += 0. if self.objective is None else self.objective_weight * self.objective(
                prediction, actuation
        )

        cost /= self.horizon

        cost += self.final_weight * (error[ -1 ] @ self.pose_weight_matrix[ -1 ] @ error[ -1 ].T).sum()

        if self.record:
            self.predicted_trajectories.append( predicted_trajectory )
            self.candidate_actuations.append( actuation )

        if cost < self.best_cost:
            self.best_candidate = candidate.copy()

        return cost

    def predict( self, actuation: ndarray ) -> ndarray:
        """
        predicts the trajectory given the proposed actuation over the horizon

        ASSUMES THAT THE MODELS STATE IS X = [POSE, POSE_DERIVATIVE]

        Parameters
        ----------
        actuation: ndarray 
            proposed actuation over the horizon
        
        Returns
        -------
        ndarray:
            predicted trajectory given the proposed actuation over the horizon; shape = (horizon, 1, state_size)
        """
        raise NotImplementedError( 'predict method should have been implemented in __init__' )

    def get_actuation( self, candidate: ndarray ) -> tuple:
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
        raise NotImplementedError( 'predict method should have been implemented in __init__' )

    def get_objective( self ) -> float:
        actuation, _ = self.get_actuation( self.raw_result.x )
        prediction = self.predict( actuation )
        return self.objective( prediction, actuation )

    def _predict_non_linear( self, actuation: ndarray ) -> ndarray:
        if self.use_prediction_cache:
            current_state = self.model.state
            key = (current_state.tobytes() + actuation.tobytes()).hex()
            cache = self.prediction_cache
            if key in cache:
                return cache[ key ]

        state = self.model.state.copy()
        predicted_trajectory = zeros( (self.horizon, 1, self.model.state.shape[ 0 ]) )

        for i in range( self.horizon ):
            state += self.model.dynamics(
                    state, actuation[ i, 0 ], self.model.perturbation
            ) * self.time_step
            predicted_trajectory[ i ] = state

        if self.use_prediction_cache:
            cache[ key ] = predicted_trajectory

        return predicted_trajectory

    def _predict_linear( self, actuation: ndarray ) -> ndarray:
        raise NotImplementedError( 'not implemented yet.' )

    def _get_actuation_from_derivative( self, candidate: ndarray ) -> tuple:
        actuation_derivatives = candidate.reshape( self.result_shape )
        actuation = actuation_derivatives.cumsum( axis=0 ) * self.time_step + self.model.actuation
        actuation = actuation.repeat( self.time_steps_per_actuation, axis=0 )
        actuation = actuation[ :self.horizon ]

        return actuation, actuation_derivatives

    def _get_actuation_from_actual( self, candidate: ndarray ) -> tuple:
        actuation = candidate.reshape( self.result_shape )
        actuation = actuation.repeat( self.time_steps_per_actuation, axis=0 )
        actuation = actuation[ :self.horizon ]
        actuation_derivatives = diff( actuation, prepend=[ [ self.model.actuation ] ], axis=0 ) / self.time_step

        return actuation, actuation_derivatives

    def _compute_result_from_derivative( self ):
        self.result = self.raw_result.x.reshape( self.result_shape )[
                          0, 0 ] * self.model.time_step + self.model.actuation

    def _compute_result_from_actual( self ):
        self.result = self.raw_result.x.reshape( self.result_shape )[ 0, 0 ]
