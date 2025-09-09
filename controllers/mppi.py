from scipy import __version__

if int( __version__.split( '.' )[ 1 ] ) < 15:
    raise ImportError(
            f"scipy version {__version__} is lower than 1.15.0 which is the minimum required version to use this "
            f"package"
    )

from time import perf_counter

from numpy import array, average, diff, exp, eye, inf, ndarray, ones, random, zeros
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize._differentialevolution import _ConstraintWrapper
from pympc.controllers.mpc import MPC
from pympc.models.model import Model


class MPPI( MPC ):

    def __init__(
            self,
            model: Model,
            horizon: int,
            target_trajectory: ndarray,
            model_type: str = 'nonlinear',
            objective: callable = None,
            time_step_prediction_factor: int = 1,
            time_steps_per_actuation: int = 1,
            guess_from_last_solution: bool = True,
            noise_covariance: float | list[ float ] = 1.,
            number_of_rolls: int = 1000,
            sampling_factor: float = .01,
            constraints: tuple[ NonlinearConstraint | LinearConstraint | Bounds ] = None,
            pose_weight_matrix: ndarray = None,
            actuation_weight_matrix: ndarray = None,
            objective_weight: float = 0.,
            final_weight: float = 0.,
            record: bool = False,
            verbose: bool = False
    ):
        """
        PREDICTION ASSUMES THAT THE MODELS STATE IS X = [POSE, POSE_DERIVATIVE]
        :param model: model of the system
        :param horizon: prediction horizon
        :param target_trajectory: target trajectory
        :param objective: objective function, must have the following signature: f(trajectory,
        actuation)
        :param time_step_prediction_factor: multiplicator for the prediction with regard to the simulation
        :param time_steps_per_actuation: number of time steps per proposed actuation over the horizon
        :param guess_from_last_solution: whether to use the last solution as the initial guess
        :param noise_covariance: noise covariance for the rolls
        :param number_of_rolls: number of rolls for the optimization algorithm
        :param sampling_factor: sampling factor of the softmax process, 0 selects only the best candidate
        while higher values factor in other candidates considering their weight
        :param bounds: bounds for the optimization variables
        :param constraints: constraints for the optimization variables
        :param pose_weight_matrix: weight matrix for the pose error; shape: (state_dim//2,
        state_dim//2)
        :param actuation_weight_matrix: weight matrix for the actuation derivative; shape:
        (actuation_dim, actuation_dim)
        :param objective_weight: weight for the objective function
        :param final_weight: weight for the final pose error
        :param record: whether to record the computation times, predicted trajectories and candidate
        actuations
        :param verbose: whether to print the optimization results
        """

        assert time_step_prediction_factor >= 1, 'time_step_prediction_factor must be greater or equal to 1'
        assert (time_step_prediction_factor == 1) or (time_steps_per_actuation != horizon), (
                'time_step_prediction_factor may only be used in constant actuation '
                'problems, not piecewise constant actuation')
        assert isinstance( noise_covariance, float ) or len( noise_covariance ) == model.actuation.shape[
            0 ], 'noise_covariance should be a float or an ndarray of the same shape as the action vector'

        match model_type:
            case 'linear':
                self.predict = self._predict_linear
            case 'nonlinear':
                self.predict = self._predict_non_linear
            case _:
                raise ValueError( f'model_type must be one of {self.MODEL_TYPE}' )

        self.model = model
        self.horizon = horizon
        self.target_trajectory = target_trajectory
        self.objective = objective

        self.time_step_prediction_factor = time_step_prediction_factor
        self.time_step = self.model.time_step

        self.time_steps_per_actuation = time_steps_per_actuation
        self.guess_from_last_solution = guess_from_last_solution
        self.noise_covariance = noise_covariance
        self.number_of_rolls = number_of_rolls
        self.sampling_factor = sampling_factor
        if constraints is not None:
            self.constraints = [ _ConstraintWrapper( constraint, zeros( constraint.lb.shape ) ) for constraint in
                                 constraints ]
        else:
            self.constraints = constraints

        add_one = (1 if self.horizon % self.time_steps_per_actuation != 0 else 0)
        self.result_shape = (
                self.horizon // self.time_steps_per_actuation + add_one,
                1,
                self.model.actuation.shape[ 0 ]
        )

        self.raw_result = zeros( self.result_shape )
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

        self.best_cost = -inf

        self.record = record
        if self.record:
            self.predicted_trajectories = [ ]
            self.candidate_actuations = [ ]
            self.compute_times = [ ]

        self.verbose = verbose

    def step( self ):
        """
        computes the best actuation for the current state with a given horizon. records the computation
        time if record is True
        """
        if self.record:
            self.predicted_trajectories.clear()
            self.candidate_actuations.clear()
            ti = perf_counter()

        disturbances = random.normal( 0, self.noise_covariance, (self.number_of_rolls,) + self.result_shape )

        mask = ones( (self.number_of_rolls,), dtype=bool )
        if self.constraints is not None:
            for constraint in self.constraints:
                mask &= array(
                        [ constraint.violation( disturbance ).sum() == 0 for disturbance in disturbances ], dtype=bool
                )
        disturbances = disturbances[ mask ]

        costs = array( [ self.cost( disturbance ) for disturbance in disturbances ] )

        if len( costs ) == 0:
            self.compute_times.append( perf_counter() - ti )
            self.result = zeros( self.model.actuation.shape )
            return self.result

        min_cost = costs.min()
        weights = exp( (min_cost - costs) / self.sampling_factor ) / sum(
                exp(
                        (min_cost - costs) / self.sampling_factor
                )
        )

        self.raw_result = average( disturbances, axis=0, weights=weights )

        if self.record:
            self.compute_times.append( perf_counter() - ti )

        self.compute_result()
        return self.result

    def get_actuation( self, candidate: ndarray ) -> tuple[ ndarray, ndarray ]:
        actuation = candidate + self.model.actuation

        actuation = actuation.repeat( self.time_steps_per_actuation, axis=0 )
        actuation = actuation[ :self.horizon ]

        actuation_derivatives = diff( actuation, prepend=[ [ self.model.actuation ] ], axis=0 ) / self.time_step

        return actuation, actuation_derivatives

    def compute_result( self ):
        self.result = self.raw_result[ 0 ].reshape( self.model.actuation.shape ) + self.model.actuation

    def add_constraint( self, constraint: Bounds | LinearConstraint | NonlinearConstraint ):
        if self.constraints is None:
            self.constraints = (_ConstraintWrapper( constraint, zeros( self.result_shape ) ),)
        else:
            self.constraints += (_ConstraintWrapper( constraint, zeros( self.result_shape ) ),)

    def get_objective( self ):
        actuation, _ = self.get_actuation( self.raw_result )
        prediction = self.predict( actuation )
        return self.objective( prediction, actuation )
