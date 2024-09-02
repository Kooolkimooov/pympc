from copy import deepcopy
from inspect import signature
from time import perf_counter

from numpy import eye, ndarray, zeros
from scipy.optimize import Bounds, LinearConstraint, minimize, NonlinearConstraint


class Model:
	def __init__(
			self,
			dynamics: callable,
			time_step: float,
			initial_state: ndarray,
			initial_actuation: ndarray,
			kwargs = None,
			record: bool = False
			):

		if kwargs is None:
			kwargs = { }

		assert list( signature( dynamics ).parameters )[ :2 ] == [ 'state', 'actuation' ]

		self.model_dynamics = dynamics
		self.time_step = time_step

		self.kwargs = kwargs

		self.state = deepcopy( initial_state )
		self.actuation = deepcopy( initial_actuation )

		self.record = record
		if self.record:
			self.previous_states = [ deepcopy( self.state ) ]
			self.previous_actuations = [ deepcopy( self.actuation ) ]

	def dynamics( self, state: ndarray, actuation: ndarray ) -> ndarray:
		return self.model_dynamics( state, actuation, **self.kwargs )

	def step( self ):
		# euler for now
		self.state += self.dynamics( self.state, self.actuation ) * self.time_step

		if self.record:
			self.previous_states.append( deepcopy( self.state ) )
			self.previous_actuations.append( deepcopy( self.actuation ) )


class MPC:
	def __init__(
			self,
			model: Model,
			horizon: int,
			target_trajectory: ndarray,
			time_steps_per_actuation: int = 1,
			guess_from_last_solution: bool = True,
			tolerance: float = 1e-6,
			max_iter: int = 1000,
			bounds: tuple[ Bounds ] = None,
			constraints: tuple[ NonlinearConstraint ] | tuple[ LinearConstraint ] = None,
			pose_weight_matrix: ndarray = None,
			actuation_derivative_weight_matrix: ndarray = None,
			final_weight: float = 0.,
			record: bool = False,
			verbose: bool = False
			):

		self.model = model
		self.horizon = horizon
		self.target_trajectory = target_trajectory
		self.time_steps_per_actuation = time_steps_per_actuation
		self.guess_from_last_solution = guess_from_last_solution
		self.tolerance = tolerance
		self.max_iter = max_iter
		self.bounds = bounds
		self.constraints = constraints

		self.result_shape = (self.horizon // self.time_steps_per_actuation + (
				1 if self.horizon % self.time_steps_per_actuation != 0 else 0), 1,
												 self.model.actuation.shape[ 0 ])

		self.raw_result = None
		self.result = zeros( self.result_shape )

		self.pose_weight_matrix: ndarray = zeros(
				(self.horizon, self.model.state.shape[ 0 ] // 2, self.model.state.shape[ 0 ] // 2)
				)
		self.actuation_derivative_weight_matrix: ndarray = zeros(
				(self.result_shape[ 0 ], self.result_shape[ 2 ], self.result_shape[ 2 ])
				)

		if pose_weight_matrix is None:
			self.pose_weight_matrix[ : ] = eye( self.model.state.shape[ 0 ] // 2 )
		else:
			self.pose_weight_matrix[ : ] = pose_weight_matrix

		if actuation_derivative_weight_matrix is None:
			self.actuation_derivative_weight_matrix[ : ] = eye( self.result_shape[ 2 ] )
		else:
			self.actuation_derivative_weight_matrix[ : ] = actuation_derivative_weight_matrix

		self.final_weight = final_weight

		self.record = record
		if self.record:
			self.predicted_trajectories = [ ]
			self.candidate_actuations = [ ]
			self.times = [ ]

		self.verbose = verbose

	def predict( self, actuation: ndarray, with_speed = False ) -> ndarray:
		p_state = deepcopy( self.model.state )
		vec_size = (self.model.state.shape[ 0 ]) if with_speed else (self.model.state.shape[ 0 ] // 2)
		predicted_trajectory = zeros( (self.horizon, 1, vec_size) )

		for i in range( self.horizon ):
			p_state += self.model.dynamics( p_state, actuation[ i, 0 ] ) * self.model.time_step
			predicted_trajectory[ i ] = p_state[ :vec_size ]

		return predicted_trajectory

	def apply_result( self ):
		self.model.actuation += self.result[ 0, 0 ]

	def optimize( self ):

		if self.record:
			self.predicted_trajectories.clear()
			self.candidate_actuations.clear()
			ti = perf_counter()

		self.raw_result = minimize(
				fun = self.cost,
				x0 = self.result.flatten(),
				tol = self.tolerance,
				bounds = self.bounds,
				constraints = self.constraints,
				options = {
						'maxiter': self.max_iter
						}
				)

		if self.record:
			self.times.append( perf_counter() - ti )

		if self.verbose:
			print( self.raw_result.message )

		self.result = self.raw_result.x.reshape( self.result_shape )

	def cost( self, actuations_derivative: ndarray ) -> float:
		actuations_derivative = actuations_derivative.reshape( self.result_shape )
		actuations = actuations_derivative.cumsum( axis = 0 ) + self.model.actuation

		actuations = actuations.repeat( self.time_steps_per_actuation, axis = 0 )
		actuations = actuations[ :self.horizon ]

		cost = 0.

		predicted_trajectory = self.predict( actuations )
		error = predicted_trajectory - self.target_trajectory
		cost += (error @ self.pose_weight_matrix @ error.transpose( (0, 2, 1) )).sum()
		cost += (
				actuations_derivative @ self.actuation_derivative_weight_matrix @
				actuations_derivative.transpose(
				(0, 2, 1)
				)).sum()

		cost /= self.horizon

		cost += self.final_weight * (error[ -1 ] @ self.pose_weight_matrix[ -1 ] @ error[ -1 ].T)

		if self.record:
			self.predicted_trajectories.append( predicted_trajectory )
			self.candidate_actuations.append( actuations )

		return cost


class Logger:
	def __init__( self ):
		self.logs: str = ''

	def log( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a tabulation
		:return: None
		'''
		print( log, end = '\t' )
		self.logs += log
		self.logs += '\t'

	def lognl( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a new line
		:return: None
		'''
		print( log )
		self.logs += log
		self.logs += '\n'

	def logrl( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a return to the beginning of the line,
		the saved text goes to a new line
		:return: None
		'''
		print( log, end = '\r' )
		self.logs += log
		self.logs += '\n'

	def save_at( self, path: str, file: str = 'logs' ):
		"""
		:param path: folder in which to save the current log
		:param file: name of the file
		"""
		with open( f'{path}/logs.txt', 'w' ) as f:
			f.write( self.logs )


def generate_trajectory(
		key_frames: list[ tuple[ float, list ] ], n_points: int
		):
	assert key_frames[ 0 ][ 0 ] == 0., "trajectory doesn't start at t = 0."

	n_dim = len( key_frames[ 0 ][ 1 ] )
	timespan = key_frames[ -1 ][ 0 ]
	trajectory = zeros( (n_points, 1, n_dim) )
	start_point = 0

	for frame_index in range( len( key_frames ) - 1 ):
		frame_0 = key_frames[ frame_index ]
		frame_1 = key_frames[ frame_index + 1 ]
		sub_timespan = frame_1[ 0 ] - frame_0[ 0 ]
		sub_n_points = int( n_points * sub_timespan / timespan )

		funcs = [ ]
		for dim in range( n_dim ):
			funcs += [ cubic_interpolation_function( frame_0[ 1 ][ dim ], frame_1[ 1 ][ dim ], 0., 0. ) ]

		for point in range( sub_n_points ):
			for dim in range( n_dim ):
				trajectory[ start_point + point, :, dim ] = funcs[ dim ]( point / sub_n_points )

		start_point += sub_n_points
	for dim in range( n_dim ):
		trajectory[ -1, :, dim ] = key_frames[ -1 ][ 1 ][ dim ]
	return trajectory


def cubic_interpolation_function( f_0: float, f_1: float, f_0p: float, f_1p: float ):
	a = 2 * f_0 - 2 * f_1 + f_0p + f_1p
	b = -3 * f_0 + 3 * f_1 - 2 * f_0p - f_1p
	c = f_0p
	d = f_0

	def function( x: float ) -> float:
		return a * pow( x, 3 ) + b * pow( x, 2 ) + c * x + d

	return function


def serialize_others( obj: any ) -> str:
	if callable( obj ):
		return obj.__name__
	if isinstance( obj, ndarray ):
		return obj.tolist()
	if isinstance( obj, NonlinearConstraint ):
		return obj.__dict__
