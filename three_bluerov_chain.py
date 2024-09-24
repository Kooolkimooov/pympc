from json import dump
from time import time

from numpy import array, cos, diff, inf, pi, r_
from numpy.linalg import norm

from bluerov import Bluerov, Bluerov3DoA
from calc_catenary_from_ext_points import *
from mpc import *
from utils import check, generate_trajectory, Logger, serialize_others


class ChainOf3( Bluerov ):
	state_size = 36
	actuation_size = 18

	br0_pose = slice( 0, 6 )
	br0_position = slice( 0, 3 )
	br0_xy = slice( 0, 2 )
	br0_z = 2
	br0_orientation = slice( 3, 6 )

	br1_pose = slice( 6, 12 )
	br1_position = slice( 6, 9 )
	br1_xy = slice( 6, 8 )
	br1_z = 8
	br1_orientation = slice( 9, 12 )

	br2_pose = slice( 12, 18 )
	br2_position = slice( 12, 15 )
	br2_xy = slice( 12, 14 )
	br2_z = 14
	br2_orientation = slice( 15, 18 )

	br0_speed = slice( 18, 24 )
	br0_linear_speed = slice( 18, 21 )
	br0_angular_speed = slice( 21, 24 )

	br1_speed = slice( 24, 30 )
	br1_linear_speed = slice( 24, 27 )
	br1_angular_speed = slice( 27, 30 )

	br2_speed = slice( 30, 36 )
	br2_linear_speed = slice( 30, 33 )
	br2_angular_speed = slice( 33, 36 )

	br0_actuation_start = 0
	br0_actuation = slice( 0, 6 )
	br0_linear_actuation = slice( 0, 3 )
	br0_angular_actuation = slice( 3, 6 )

	br1_actuation_start = 6
	br1_actuation = slice( 6, 12 )
	br1_linear_actuation = slice( 6, 9 )
	br1_angular_actuation = slice( 9, 12 )

	br2_actuation_start = 12
	br2_actuation = slice( 12, 18 )
	br2_linear_actuation = slice( 12, 15 )
	br2_angular_actuation = slice( 15, 18 )

	br0_state = r_[ br0_pose, br0_speed ]
	br1_state = r_[ br1_pose, br1_speed ]
	br2_state = r_[ br2_pose, br2_speed ]

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		"""
		evalutes the dynamics of each robot of the chain
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""
		state_derivative = zeros( state.shape )

		state_derivative[ self.br0_state ] = Bluerov.__call__(
				self, state[ self.br0_state ], actuation[ self.br0_actuation ]
				)
		state_derivative[ self.br1_state ] = Bluerov.__call__(
				self, state[ self.br1_state ], actuation[ self.br1_actuation ]
				)
		state_derivative[ self.br2_state ] = Bluerov.__call__(
				self, state[ self.br2_state ], actuation[ self.br2_actuation ]
				)

		return state_derivative


class ChainOf33DoA( ChainOf3, Bluerov3DoA ):
	actuation_size = 9

	br0_actuation_start = 0
	br0_actuation = slice( 0, 3 )
	br0_linear_actuation = slice( 0, 3 )
	br0_angular_actuation = slice( 0, 0 )

	br1_actuation_start = 3
	br1_actuation = slice( 3, 6 )
	br1_linear_actuation = slice( 3, 6 )
	br1_angular_actuation = slice( 0, 0 )

	br2_actuation_start = 6
	br2_actuation = slice( 6, 9 )
	br2_linear_actuation = slice( 6, 9 )
	br2_angular_actuation = slice( 0, 0 )

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		"""
		evalutes the dynamics of each robot of the chain
		:param state: current state of the system
		:param actuation: current actuation of the system
		:return: state derivative of the system
		"""
		state_derivative = zeros( state.shape )

		state_derivative[ self.br0_state ] = Bluerov3DoA.__call__(
				self, state[ self.br0_state ], actuation[ self.br0_actuation ]
				)
		state_derivative[ self.br1_state ] = Bluerov3DoA.__call__(
				self, state[ self.br1_state ], actuation[ self.br1_actuation ]
				)
		state_derivative[ self.br2_state ] = Bluerov3DoA.__call__(
				self, state[ self.br2_state ], actuation[ self.br2_actuation ]
				)

		return state_derivative


def three_robot_chain_objective( trajectory: ndarray, actuation: ndarray ):
	objective = 0.

	trajectory_derivative = diff( trajectory, axis = 0 )
	objective += norm( trajectory_derivative[ :, 0, ChainOf3.br0_position ], axis = 1 ).sum()
	objective += norm( trajectory_derivative[ :, 0, ChainOf3.br1_position ], axis = 1 ).sum()
	objective += norm( trajectory_derivative[ :, 0, ChainOf3.br2_position ], axis = 1 ).sum()

	return objective


def three_robot_chain_constraint( candidate_actuation_derivative ):
	global three_bluerov_chain_mpc

	candidate_actuation = candidate_actuation_derivative.reshape(
			three_bluerov_chain_mpc.result_shape
			).cumsum( axis = 0 ) * three_bluerov_chain_mpc.time_step + three_bluerov_chain_mpc.model.actuation
	candidate_actuation = candidate_actuation.repeat(
			three_bluerov_chain_mpc.time_steps_per_actuation, axis = 0
			)

	predicted_trajectory = three_bluerov_chain_mpc.predict( candidate_actuation, with_speed = True )
	predicted_trajectory = predicted_trajectory[ :, 0 ]

	# 2 constraints on cables (lowest points),
	# 4 on inter robot_distance (2 horizontal, 2 vertical),
	# 3 on robot speed
	n_constraints = 2 + 4 + 3
	constraint = zeros( (three_bluerov_chain_mpc.horizon, n_constraints) )

	# z position of robots 0 and 1; we add H afterward
	constraint[ :, 0 ] = predicted_trajectory[ :, ChainOf3.br0_z ]
	constraint[ :, 1 ] = predicted_trajectory[ :, ChainOf3.br1_z ]

	# horizontal distance between consecutive robots
	constraint[ :, 2 ] = norm(
			predicted_trajectory[ :, ChainOf3.br1_xy ] - predicted_trajectory[ :, ChainOf3.br0_xy ], axis = 1
			)
	constraint[ :, 3 ] = norm(
			predicted_trajectory[ :, ChainOf3.br2_xy ] - predicted_trajectory[ :, ChainOf3.br1_xy ], axis = 1
			)

	# distance between consecutive robots
	constraint[ :, 4 ] = norm(
			predicted_trajectory[ :, ChainOf3.br1_position ] -

			predicted_trajectory[ :, ChainOf3.br0_position ], axis = 1
			)
	constraint[ :, 5 ] = norm(
			predicted_trajectory[ :, ChainOf3.br2_position ] -

			predicted_trajectory[ :, ChainOf3.br1_position ], axis = 1
			)

	# speed
	constraint[ :, 6 ] = norm( predicted_trajectory[ :, ChainOf3.br0_linear_speed ], axis = 1 )
	constraint[ :, 7 ] = norm( predicted_trajectory[ :, ChainOf3.br1_linear_speed ], axis = 1 )
	constraint[ :, 8 ] = norm( predicted_trajectory[ :, ChainOf3.br2_linear_speed ], axis = 1 )

	for i, state in enumerate( predicted_trajectory ):

		try:
			_, _, H01 = get_catenary_param(
					state[ ChainOf3.br0_z ] - state[ ChainOf3.br1_z ], constraint[ i, 2 ], 3
					)
			_, _, H12 = get_catenary_param(
					state[ ChainOf3.br1_z ] - state[ ChainOf3.br2_z ], constraint[ i, 3 ], 3
					)
		except:
			H01 = 1.5
			H12 = 1.5

		constraint[ i, 0 ] += H01
		constraint[ i, 1 ] += H12

	return constraint.flatten()


if __name__ == "__main__":

	ti = perf_counter()

	n_frames = 2000
	tolerance = 1e-3
	time_step = 0.01

	state = zeros( (ChainOf3.state_size,) )
	state[ ChainOf3.br0_position.start ] = 2.
	state[ ChainOf3.br1_position.start ] = 2.5
	state[ ChainOf3.br2_position.start ] = 3.

	actuation = zeros( (ChainOf3.actuation_size,) )

	horizon = 25
	time_steps_per_act = 25

	key_frames = [ (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 12), (.5, [ -3., 0., 0., 0., 0., 0. ] + [ 0. ] * 12),
								 (1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 12), (2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 12) ]

	trajectory = generate_trajectory( key_frames, 2 * n_frames )
	trajectory[ :, 0, 2 ] = 1.5 * cos( 1.25 * (trajectory[ :, 0, 0 ] - 2) + pi ) + 1.5

	max_required_speed = (max( norm( diff( trajectory[ :, 0, :3 ], axis = 0 ), axis = 1 ) ) / time_step)
	print( f'{max_required_speed=}' )

	# plt.plot( trajectory[:, 0, 2] )
	# plt.plot( norm(diff(trajectory[:, 0, :3], axis=0), axis=1) / time_step )
	# plt.show()
	# exit()

	pose_weight_matrix = eye( actuation.shape[ 0 ] )
	pose_weight_matrix[ ChainOf3.br0_position, ChainOf3.br0_position ] *= 10.
	pose_weight_matrix[ ChainOf3.br0_orientation, ChainOf3.br0_orientation ] *= 5.
	pose_weight_matrix[ ChainOf3.br1_position, ChainOf3.br1_position ] *= 0.
	pose_weight_matrix[ ChainOf3.br1_orientation, ChainOf3.br1_orientation ] *= 5.
	pose_weight_matrix[ ChainOf3.br2_position, ChainOf3.br2_position ] *= 0.
	pose_weight_matrix[ ChainOf3.br2_orientation, ChainOf3.br2_orientation ] *= 5.

	actuation_weight_matrix = eye( actuation.shape[ 0 ] )
	actuation_weight_matrix[ ChainOf3.br0_linear_actuation, ChainOf3.br0_linear_actuation ] *= 0.
	actuation_weight_matrix[ ChainOf3.br0_angular_actuation, ChainOf3.br0_angular_actuation ] *= 1000.
	actuation_weight_matrix[ ChainOf3.br1_linear_actuation, ChainOf3.br1_linear_actuation ] *= 0.
	actuation_weight_matrix[ ChainOf3.br1_angular_actuation, ChainOf3.br1_angular_actuation ] *= 1000.
	actuation_weight_matrix[ ChainOf3.br2_linear_actuation, ChainOf3.br2_linear_actuation ] *= 0.
	actuation_weight_matrix[ ChainOf3.br2_angular_actuation, ChainOf3.br2_angular_actuation ] *= 1000.

	final_cost_weight = 10.

	three_bluerov_chain_model = Model(
			ChainOf3(), time_step, state, actuation, record = True
			)

	three_bluerov_chain_mpc = MPC(
			three_bluerov_chain_model,
			horizon,
			trajectory,
			objective = three_robot_chain_objective,
			objective_weight = 10.,
			tolerance = tolerance,
			time_steps_per_actuation = time_steps_per_act,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True
			)

	# mpc_controller.verbose = True

	floor_depth = 3.00001
	du_l_lb = -20.
	du_l_ub = 20.
	du_a_lb = -.1
	du_a_ub = .1
	dp_lb = 0.4
	dp_ub = 2.8
	dr_lb = -inf
	dr_ub = 2.8
	v_lb = -inf
	v_ub = 3.

	three_bluerov_chain_mpc.bounds = Bounds(
			array( [ [ du_l_lb, du_l_lb, du_l_lb, du_a_lb, du_a_lb, du_a_lb ] ] ).repeat( 3, axis = 0 ).flatten(),
			array( [ [ du_l_ub, du_l_ub, du_l_ub, du_a_ub, du_a_ub, du_a_ub ] ] ).repeat( 3, axis = 0 ).flatten()
			)

	constraints_labels = [ '$z_0+H_{01}$', '$z_1+H_{12}$', '$|P_0^{x,y}-P_1^{x,y}|$', '$|P_1^{x,y}-P_2^{x,y}|$',
												 '$|P_0^{x,y,z}-P_1^{x,y,z}|$', '$|P_1^{x,y,z}-P_2^{x,y,z}|$', '$|V_0|$', '$|V_1|$',
												 '$|V_2|$' ]

	lb_base = [ -inf, -inf, dp_lb, dp_lb, dr_lb, dr_lb, v_lb, v_lb, v_lb ]
	ub_base = [ floor_depth, floor_depth, dp_ub, dp_ub, dr_ub, dr_ub, v_ub, v_ub, v_ub ]

	lb = [ lb_base ] * horizon
	ub = [ ub_base ] * horizon

	constraint = NonlinearConstraint( three_robot_chain_constraint, array( lb ).flatten(), array( ub ).flatten() )
	constraint.labels = constraints_labels

	three_bluerov_chain_mpc.constraints = (constraint,)

	previous_nfeval_record = [ 0 ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]

	logger = Logger()

	folder = f'./export/three_robots_chain_{int( time() )}'
	if not (check( folder ) and check( f'{folder}/data' )):
		exit()

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( three_bluerov_chain_mpc.__dict__, f, default = serialize_others )

	for frame in range( n_frames ):

		three_bluerov_chain_mpc.target_trajectory = trajectory[ frame + 1:frame + horizon + 1 ]

		three_bluerov_chain_mpc.compute_actuation()
		three_bluerov_chain_mpc.apply_result()
		three_bluerov_chain_model.step()

		# try to recover
		if not three_bluerov_chain_mpc.raw_result.success and three_bluerov_chain_mpc.tolerance < 1:
			three_bluerov_chain_mpc.tolerance *= 10
		elif (three_bluerov_chain_mpc.raw_result.success and three_bluerov_chain_mpc.tolerance > tolerance):
			three_bluerov_chain_mpc.tolerance /= 10

		with open( f'{folder}/data/{frame}.json', 'w' ) as f:
			dump( three_bluerov_chain_mpc.__dict__, f, default = serialize_others )

		logger.log( f"{frame}/{n_frames}" )
		logger.log( f"{perf_counter() - ti:.6f}" )
		logger.log( f"{len( three_bluerov_chain_model.previous_states )}" )
		logger.log( f"{len( three_bluerov_chain_mpc.predicted_trajectories )}" )
		logger.log( f"{three_bluerov_chain_mpc.compute_times[ -1 ]:.6f}" )
		logger.log( f"{three_bluerov_chain_mpc.tolerance} - {three_bluerov_chain_mpc.raw_result.success}" )

		logger.lognl( "" )
		logger.save_at( folder )
