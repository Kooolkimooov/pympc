from glob import glob
from json import dump
from os import mkdir, path, remove
from time import time

from cycler import cycler
from numpy import array, cos, cross, diag, diff, inf, meshgrid, nan, ones, pi, sin, sqrt, tan
from numpy.linalg import inv, norm
from PIL import Image
from scipy.spatial.transform import Rotation

from calc_catenary_from_ext_points import *
from mpc import *


def compute_rotation_matrix( phi: float, theta: float, psi: float ) -> ndarray:
	cPhi, sPhi = cos( phi ), sin( phi )
	cTheta, sTheta, tTheta = cos( theta ), sin( theta ), tan( theta )
	cPsi, sPsi = cos( psi ), sin( psi )
	matrix = zeros( (6, 6) )
	matrix[ 0, :3 ] = array(
			[ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi, sPsi * sPhi + cPsi * sTheta * cPhi ]
			)
	matrix[ 1, :3 ] = array(
			[ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi, -cPsi * sPhi + sPsi * sTheta * cPhi ]
			)
	matrix[ 2, :3 ] = array( [ -sTheta, cTheta * sPhi, cTheta * cPhi ] )
	matrix[ 3, 3: ] = array( [ 1, sPhi * tTheta, cPhi * tTheta ] )
	matrix[ 4, 3: ] = array( [ 0, cPhi, -sPhi ] )
	matrix[ 5, 3: ] = array( [ 0, sPhi / cTheta, cPhi / cTheta ] )
	return matrix


def build_inertial_matrix(
		mass: float, center_of_mass: ndarray, inertial_coefficients: list[ float ]
		):
	inertial_matrix = eye( 6 )
	for i in range( 3 ):
		inertial_matrix[ i, i ] = mass
		inertial_matrix[ i + 3, i + 3 ] = inertial_coefficients[ i ]
	inertial_matrix[ 0, 4 ] = mass * center_of_mass[ 2 ]
	inertial_matrix[ 0, 5 ] = - mass * center_of_mass[ 1 ]
	inertial_matrix[ 1, 3 ] = - mass * center_of_mass[ 2 ]
	inertial_matrix[ 1, 5 ] = mass * center_of_mass[ 0 ]
	inertial_matrix[ 2, 3 ] = mass * center_of_mass[ 1 ]
	inertial_matrix[ 2, 4 ] = - mass * center_of_mass[ 0 ]
	inertial_matrix[ 4, 0 ] = mass * center_of_mass[ 2 ]
	inertial_matrix[ 5, 0 ] = - mass * center_of_mass[ 1 ]
	inertial_matrix[ 3, 1 ] = - mass * center_of_mass[ 2 ]
	inertial_matrix[ 5, 1 ] = mass * center_of_mass[ 0 ]
	inertial_matrix[ 3, 2 ] = mass * center_of_mass[ 1 ]
	inertial_matrix[ 4, 2 ] = - mass * center_of_mass[ 0 ]
	inertial_matrix[ 3, 4 ] = - inertial_coefficients[ 3 ]
	inertial_matrix[ 3, 5 ] = - inertial_coefficients[ 4 ]
	inertial_matrix[ 4, 5 ] = - inertial_coefficients[ 5 ]
	inertial_matrix[ 4, 3 ] = - inertial_coefficients[ 3 ]
	inertial_matrix[ 5, 3 ] = - inertial_coefficients[ 4 ]
	inertial_matrix[ 5, 4 ] = - inertial_coefficients[ 5 ]

	return inertial_matrix


def compute_hydrostatic_force(
		weight: ndarray,
		buoyancy: ndarray,
		center_of_mass: ndarray,
		center_of_volume: ndarray,
		rotation_matrix
		) -> ndarray:
	force = zeros( 6 )
	force[ :3 ] = rotation_matrix.T @ (weight + buoyancy)
	force[ 3: ] = cross( center_of_mass, rotation_matrix.T @ weight ) + cross(
			center_of_volume, rotation_matrix.T @ buoyancy
			)

	return force


def three_robots_chain_follow(
		state: ndarray,
		actuation: ndarray,
		weight: ndarray,
		buoyancy: ndarray,
		center_of_mass: ndarray,
		center_of_volume: ndarray,
		inverted_inertial_matrix: ndarray,
		hydrodynamic_matrix: ndarray
		) -> ndarray:
	"""
	:param state: state of the chain such that
	-----0:6--------------6:12-------------12:18------------18:24-------------24:30-------------30
	:36-----
	x = [pose_robot_0_wf, pose_robot_1_wf, pose_robot_2_wf, vel_robot_0_rf,	vel_robot_1_rf,
	vel_robot_2_rf]
	:param actuation: actuation of the chain such that u = [actuation_robot_0, actuation_robot_1]
	:param hydrodynamic_matrix:
	:param inverted_inertial_matrix:
	:param center_of_volume:
	:param center_of_mass:
	:param buoyancy:
	:param weight:
	:return: xdot: derivative of the state of the chain
	"""
	x0 = state[ :6 ]
	x0d = state[ 18:24 ]

	x1 = state[ 6:12 ]
	x1d = state[ 24:30 ]
	u1 = actuation[ 0:6 ]

	x2 = state[ 12:18 ]
	x2d = state[ 30:36 ]
	u2 = actuation[ 6:12 ]

	R0 = compute_rotation_matrix( *x0[ 3: ] )
	R1 = compute_rotation_matrix( *x1[ 3: ] )
	R2 = compute_rotation_matrix( *x2[ 3: ] )

	s0 = compute_hydrostatic_force(
			weight, buoyancy, center_of_mass, center_of_volume, R0[ :3, :3 ]
			)

	s1 = compute_hydrostatic_force(
			weight, buoyancy, center_of_mass, center_of_volume, R1[ :3, :3 ]
			)
	s2 = compute_hydrostatic_force(
			weight, buoyancy, center_of_mass, center_of_volume, R2[ :3, :3 ]
			)

	xdot = zeros( state.shape )

	xdot[ :6 ] = R0 @ x0d
	xdot[ 18:24 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x0d + s0)

	xdot[ 6:12 ] = R1 @ x1d
	xdot[ 24:30 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x1d + s1 + u1)

	xdot[ 12:18 ] = R2 @ x2d
	xdot[ 30:36 ] = inverted_inertial_matrix @ (hydrodynamic_matrix @ x2d + s2 + u2)

	return xdot


def constraint_f( candidate_actuation_derivative ):
	global mpc_controller

	candidate_actuation = candidate_actuation_derivative.reshape(
			mpc_controller.result_shape
			).cumsum( axis = 0 ) + mpc_controller.model.actuation
	candidate_actuation = candidate_actuation.repeat(
			mpc_controller.time_steps_per_actuation, axis = 0
			)

	predicted_trajectory = mpc_controller.predict( candidate_actuation, with_speed = True )

	# 2 constraints on cables (lowest points),
	# 4 on inter robot_distance (2 horizontal, 2 vertical),
	# 4 on robot speed
	n_constraints = 2 + 4 + 4
	constraint = zeros( (mpc_controller.horizon, n_constraints) )

	for i, state in enumerate( predicted_trajectory[ :, 0 ] ):

		dp01 = norm( state[ 6:8 ] - state[ 0:2 ] )
		dz01 = -(state[ 8 ] - state[ 2 ])
		dp12 = norm( state[ 12:14 ] - state[ 6:8 ] )
		dz12 = -(state[ 14 ] - state[ 8 ])

		try:
			_, _, H01 = get_catenary_param( dz01, dp01, 3 )
			_, _, H12 = get_catenary_param( dz12, dp12, 3 )
		except:
			H01 = 1.5
			H12 = 1.5

		constraint[ i, 0 ] = state[ 2 ] + H01
		constraint[ i, 1 ] = state[ 8 ] + H12
		constraint[ i, 2 ] = norm( state[ 6:9 ] - state[ 0:3 ] )
		constraint[ i, 3 ] = norm( state[ 12:15 ] - state[ 0:3 ] )
		constraint[ i, 4 ] = abs( dz01 )
		constraint[ i, 5 ] = abs( dz12 )
		constraint[ i, 6 ] = norm( state[ 24:27 ] )
		constraint[ i, 7 ] = norm( state[ 27:30 ] )
		constraint[ i, 8 ] = norm( state[ 30:33 ] )
		constraint[ i, 9 ] = norm( state[ 33:36 ] )

	return constraint.flatten()


def chain_objective( trajectory: ndarray, actuation: ndarray ) -> float:
	objective = 0.

	objective += norm( trajectory[ :, 0, 0:3 ] - trajectory[ :, 0, 6:9 ] - 1. ) ** 2
	objective += norm( trajectory[ :, 0, 6:9 ] - trajectory[ :, 0, 12:15 ] - 1. ) ** 2

	return objective


def plot(
		mpc: MPC,
		full_trajectory: ndarray,
		c_frame: int,
		n_frame: int,
		volume: ndarray,
		max_ux: float,
		max_ut: float,
		floor_z: float,
		f_eval_record: list[ int ],
		H01_record: list[ float ],
		H12_record: list[ float ]
		):
	# we record the initial value + the new value after the integration in `step()`
	time_previous = [ i * mpc.model.time_step - (c_frame + 1) * mpc.model.time_step for i in
										range( len( mpc.model.previous_states ) ) ]
	time_prediction = [ i * mpc.model.time_step for i in
											range( mpc.predicted_trajectories[ 0 ].shape[ 0 ] - 1 ) ]

	fig_grid_shape = (15, 22)
	vsize = 13
	psizex = 3
	psizey = 3
	r0x = vsize
	r1x = vsize + psizex
	r2x = vsize + 2 * psizex
	figure = plt.figure( figsize = [ 21, 9 ] )

	view = plt.subplot2grid( fig_grid_shape, (0, 0), vsize, vsize, figure, projection = '3d' )
	view.set_xlabel( "x" )
	view.set_ylabel( "y" )
	view.set_zlabel( "z" )
	view.set_xlim( *volume[ 0 ] )
	view.set_ylim( *volume[ 1 ] )
	view.set_zlim( *volume[ 2 ] )
	view.invert_yaxis()
	view.invert_zaxis()

	ax_pos0 = plt.subplot2grid( fig_grid_shape, (0, r0x), psizey, psizex, figure )
	ax_pos0.set_title( 'robot0' )
	ax_pos0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_pos0.set_ylim( min( volume[ :, 0 ] ), max( volume[ :, 1 ] ) )
	ax_pos0.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_pos0.yaxis.set_label_position( "right" )
	ax_pos0.yaxis.tick_right()

	ax_ang0 = plt.subplot2grid( fig_grid_shape, (psizey, r0x), psizey, psizex, figure )
	ax_ang0.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_ang0.set_ylim( - pi, pi )
	ax_ang0.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_ang0.yaxis.set_label_position( "right" )
	ax_ang0.yaxis.tick_right()

	ax_cat_H = plt.subplot2grid(
			fig_grid_shape, (4 * psizey + 1, r1x - psizex // 2), psizey - 1, 5, figure
			)
	ax_cat_H.set_xlabel( 'time' )
	ax_cat_H.set_ylabel( 'lowest point' )
	ax_cat_H.set_xlim( time_previous[ 0 ], time_previous[ -1 ] )
	ax_cat_H.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

	ax_pos1 = plt.subplot2grid( fig_grid_shape, (0, r1x), psizey, psizex, figure )
	ax_pos1.set_title( 'robot1' )
	ax_pos1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_pos1.set_ylim( min( volume[ :, 0 ] ), max( volume[ :, 1 ] ) )
	ax_pos1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_pos1.yaxis.set_label_position( "right" )
	ax_pos1.yaxis.tick_right()

	ax_ang1 = plt.subplot2grid( fig_grid_shape, (psizey, r1x), psizey, psizex, figure )
	ax_ang1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_ang1.set_ylim( - pi, pi )
	ax_ang1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_ang1.yaxis.set_label_position( "right" )
	ax_ang1.yaxis.tick_right()

	ax_act_pos1 = plt.subplot2grid( fig_grid_shape, (2 * psizey, r1x), psizey, psizex, figure )
	ax_act_pos1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_act_pos1.set_ylim( - max_ux, max_ux )
	ax_act_pos1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_act_pos1.yaxis.set_label_position( "right" )
	ax_act_pos1.yaxis.tick_right()

	ax_act_ang1 = plt.subplot2grid( fig_grid_shape, (3 * psizey, r1x), psizey, psizex, figure )
	ax_act_ang1.set_xlabel( 'time' )
	ax_act_ang1.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_act_ang1.set_ylim( -max_ut, max_ut )
	ax_act_ang1.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_act_ang1.yaxis.set_label_position( "right" )
	ax_act_ang1.yaxis.tick_right()

	ax_pos2 = plt.subplot2grid( fig_grid_shape, (0, r2x), psizey, psizex, figure )
	ax_pos2.set_title( 'robot2' )
	ax_pos2.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_pos2.set_ylim( min( volume[ :, 0 ] ), max( volume[ :, 1 ] ) )
	ax_pos2.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_pos2.set_ylabel( 'position' )
	ax_pos2.yaxis.set_label_position( "right" )
	ax_pos2.yaxis.tick_right()

	ax_ang2 = plt.subplot2grid( fig_grid_shape, (psizey, r2x), psizey, psizex, figure )
	ax_ang2.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_ang2.set_ylim( -pi, pi )
	ax_ang2.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_ang2.set_ylabel( 'orientation' )
	ax_ang2.yaxis.set_label_position( "right" )
	ax_ang2.yaxis.tick_right()

	ax_act_pos2 = plt.subplot2grid( fig_grid_shape, (2 * psizey, r2x), psizey, psizex, figure )
	ax_act_pos2.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_act_pos2.set_ylim( - max_ux, max_ux )
	ax_act_pos2.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_act_pos2.set_ylabel( 'pos. act.' )
	ax_act_pos2.yaxis.set_label_position( "right" )
	ax_act_pos2.yaxis.tick_right()

	ax_act_ang2 = plt.subplot2grid( fig_grid_shape, (3 * psizey, r2x), psizey, psizex, figure )
	ax_act_ang2.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_act_ang2.set_ylim( - max_ut, max_ut )
	ax_act_ang2.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )
	ax_act_ang2.set_ylabel( 'ang. act.' )
	ax_act_ang2.yaxis.set_label_position( "right" )
	ax_act_ang2.yaxis.tick_right()

	ax_comp_time = plt.subplot2grid( fig_grid_shape, (vsize, 1), 2, 5, figure )
	ax_comp_time.set_prop_cycle( cycler( 'color', [ 'blue', 'red' ] ) )
	ax_comp_time.set_ylabel( 'comp. time' )
	ax_comp_time.set_xlabel( 'time' )
	ax_comp_time.set_xlim( time_previous[ 0 ], time_previous[ -1 ] )

	ax_nfeval = plt.subplot2grid( fig_grid_shape, (vsize, 7), 2, 5, figure )
	ax_nfeval.set_prop_cycle( cycler( 'color', [ 'blue', 'red' ] ) )
	ax_nfeval.set_ylabel( 'n. eval' )
	ax_nfeval.set_xlabel( 'time' )
	ax_nfeval.set_xlim( time_previous[ 0 ], time_previous[ -1 ] )

	plt.subplots_adjust( hspace = 0., wspace = 0. )
	figure.suptitle(
			f"{c_frame + 1}/{n_frame} - {mpc.times[ -1 ]}s - {len(mpc.predicted_trajectories)=}"
			)

	target_pose = mpc.target_trajectory[ 0, 0 ]

	state_r0 = Rotation.from_euler( 'xyz', mpc.model.state[ 3:6 ] ).as_matrix()
	state_r1 = Rotation.from_euler( 'xyz', mpc.model.state[ 9:12 ] ).as_matrix()
	state_r2 = Rotation.from_euler( 'xyz', mpc.model.state[ 15:18 ] ).as_matrix()
	target_r0 = Rotation.from_euler( 'xyz', target_pose[ 3:6 ] ).as_matrix()

	surf_x = volume[ 0 ]
	surf_y = volume[ 1 ]
	surf_x, surf_y = meshgrid( surf_x, surf_y )
	surf_z = ones( surf_x.shape ) * floor_z
	view.plot_surface( surf_x, surf_y, surf_z, alpha = 0.1 )

	quiver_scale = .25
	view.quiver(
			*mpc.model.state[ :3 ], *(state_r0 @ (quiver_scale * array( [ 1., 0., 0. ] ))),
			color = 'blue'
			)
	view.quiver(
			*mpc.model.state[ 6:9 ], *(state_r1 @ (quiver_scale * array( [ 1., 0., 0. ] ))),
			color = 'red'
			)
	view.quiver(
			*mpc.model.state[ 12:15 ],
			*(state_r2 @ (quiver_scale * array( [ 1., 0., 0. ] ))),
			color = 'green'
			)
	view.quiver(
			*target_pose[ :3 ], *(target_r0 @ (quiver_scale * array( [ 1., 0., 0. ] ))), color = 'black'
			)

	previous_states_array = array( mpc.model.previous_states )

	view.plot(
			previous_states_array[ :, 0 ],
			previous_states_array[ :, 1 ],
			previous_states_array[ :, 2 ],
			color = 'blue'
			)
	view.plot(
			previous_states_array[ :, 6 ],
			previous_states_array[ :, 7 ],
			previous_states_array[ :, 8 ],
			color = 'red'
			)
	view.plot(
			previous_states_array[ :, 12 ],
			previous_states_array[ :, 13 ],
			previous_states_array[ :, 14 ],
			color = 'green'
			)

	view.plot(
			full_trajectory[ :, 0, 0 ], full_trajectory[ :, 0, 1 ], full_trajectory[ :, 0, 2 ], ':'
			)

	try:
		cat01, _, _, H01 = get_coor_marker_points_ideal_catenary(
				mpc.model.state[ 0 ],
				-mpc.model.state[ 1 ],
				-mpc.model.state[ 2 ],
				mpc.model.state[ 6 ],
				-mpc.model.state[ 7 ],
				-mpc.model.state[ 8 ],
				3.,
				.2
				)
		cat12, _, _, H12 = get_coor_marker_points_ideal_catenary(
				mpc.model.state[ 6 ],
				-mpc.model.state[ 7 ],
				-mpc.model.state[ 8 ],
				mpc.model.state[ 12 ],
				-mpc.model.state[ 13 ],
				-mpc.model.state[ 14 ],
				3.,
				.2
				)
		cat01[ :, 1: ] *= -1
		cat12[ :, 1: ] *= -1
	except:
		cat01 = array( [ mpc.model.state[ :3 ], mpc.model.state[ 6:9 ] ] )
		cat12 = array( [ mpc.model.state[ 6:9 ], mpc.model.state[ 12:15 ] ] )
		H01 = nan
		H12 = nan

	f_eval_record.append( len( mpc.predicted_trajectories ) )
	H01_record.append( H01 + mpc.model.state[ 2 ] )
	H12_record.append( H12 + mpc.model.state[ 8 ] )

	view.plot( cat01[ :, 0 ], cat01[ :, 1 ], cat01[ :, 2 ], 'blue' )
	view.plot( cat12[ :, 0 ], cat12[ :, 1 ], cat12[ :, 2 ], 'red' )

	ax_cat_H.plot( time_previous, H01_record )
	ax_cat_H.plot( time_previous, H12_record )

	ax_pos0.plot(
			time_previous + time_prediction[ 1: ],
			full_trajectory[ :len( time_previous ) + len( time_prediction ) - 1, 0, :3 ],
			':'
			)
	ax_ang0.plot(
			time_previous + time_prediction[ 1: ],
			(full_trajectory[ :len( time_previous ) + len( time_prediction ) - 1, 0, 3:6 ] + pi) % (
					2 * pi) - pi,
			':'
			)

	previous_pos_record_array0 = array( mpc.model.previous_states )[ :, :3 ]
	previous_ang_record_array0 = (array( mpc.model.previous_states )[ :, 3:6 ] + pi) % (2 * pi) - pi
	previous_act_pos_record_array0 = array( mpc.model.previous_actuations )[ :, :3 ]
	previous_act_ang_record_array0 = array( mpc.model.previous_actuations )[ :, 3:6 ]
	previous_pos_record_array1 = array( mpc.model.previous_states )[ :, 6:9 ]
	previous_ang_record_array1 = (array( mpc.model.previous_states )[ :, 9:12 ] + pi) % (2 * pi) - pi
	previous_act_pos_record_array1 = array( mpc.model.previous_actuations )[ :, 6:9 ]
	previous_act_ang_record_array1 = array( mpc.model.previous_actuations )[ :, 9:12 ]
	previous_pos_record_array2 = array( mpc.model.previous_states )[ :, 12:15 ]
	previous_ang_record_array2 = (
			(array( mpc.model.previous_states )[ :, 15:18 ] + pi) % (2 * pi) - pi)

	ax_pos0.plot( time_previous, previous_pos_record_array0 )
	ax_ang0.plot( time_previous, previous_ang_record_array0 )
	ax_pos1.plot( time_previous, previous_pos_record_array1 )
	ax_ang1.plot( time_previous, previous_ang_record_array1 )
	ax_act_pos1.plot( time_previous, previous_act_pos_record_array0 )
	ax_act_ang1.plot( time_previous, previous_act_ang_record_array0 )
	ax_pos2.plot( time_previous, previous_pos_record_array2 )
	ax_ang2.plot( time_previous, previous_ang_record_array2 )
	ax_act_pos2.plot( time_previous, previous_act_pos_record_array1 )
	ax_act_ang2.plot( time_previous, previous_act_ang_record_array1 )
	# ax_obj.plot( time_previous, previous_objective_record )

	step = 1
	if len( mpc.predicted_trajectories ) > 1000:
		step = len( mpc.predicted_trajectories ) // 1000

	for f_eval in range( 0, len( mpc.predicted_trajectories ), step ):
		pos_record_array0 = mpc.predicted_trajectories[ f_eval ][ 1:, 0, :3 ]
		ang_record_array0 = (mpc.predicted_trajectories[ f_eval ][ 1:, 0, 3:6 ] + pi) % (2 * pi) - pi
		pos_record_array1 = mpc.predicted_trajectories[ f_eval ][ 1:, 0, 6:9 ]
		ang_record_array1 = (mpc.predicted_trajectories[ f_eval ][ 1:, 0, 9:12 ] + pi) % (2 * pi) - pi
		pos_record_array2 = mpc.predicted_trajectories[ f_eval ][ 1:, 0, 12:15 ]
		ang_record_array2 = (mpc.predicted_trajectories[ f_eval ][ 1:, 0, 15:18 ] + pi) % (2 * pi) - pi

		view.plot(
				pos_record_array0[ :, 0 ],
				pos_record_array0[ :, 1 ],
				pos_record_array0[ :, 2 ],
				'blue',
				linewidth = .1
				)
		ax_pos0.plot( time_prediction, pos_record_array0, linewidth = .1 )
		ax_ang0.plot( time_prediction, ang_record_array0, linewidth = .1 )

		ax_act_pos1.plot(
				time_prediction, mpc.candidate_actuations[ f_eval ][ 1:, 0, :3 ], linewidth = .1
				)
		ax_act_ang1.plot(
				time_prediction, mpc.candidate_actuations[ f_eval ][ 1:, 0, 3:6 ], linewidth = .1
				)

		view.plot(
				pos_record_array1[ :, 0 ],
				pos_record_array1[ :, 1 ],
				pos_record_array1[ :, 2 ],
				'red',
				linewidth = .1
				)
		ax_pos1.plot( time_prediction, pos_record_array1, linewidth = .1 )
		ax_ang1.plot( time_prediction, ang_record_array1, linewidth = .1 )
		ax_act_pos2.plot(
				time_prediction, mpc.candidate_actuations[ f_eval ][ 1:, 0, 6:9 ], linewidth = .1
				)
		ax_act_ang2.plot(
				time_prediction, mpc.candidate_actuations[ f_eval ][ 1:, 0, 9:12 ], linewidth = .1
				)

		view.plot(
				pos_record_array2[ :, 0 ],
				pos_record_array2[ :, 1 ],
				pos_record_array2[ :, 2 ],
				'green',
				linewidth = .1
				)
		ax_pos2.plot( time_prediction, pos_record_array2, linewidth = .1 )
		ax_ang2.plot( time_prediction, ang_record_array2, linewidth = .1 )

	# ax_obj.plot( time_prediction, mpc_config[ 'objective_record' ][ f_eval ], linewidth = .1 )

	# plot vertical line from y min to y max
	ax_pos0.axvline( color = 'k' )
	ax_pos0.axhline( floor_z, color = 'green' )
	ax_ang0.axvline( color = 'k' )
	ax_pos1.axvline( color = 'k' )
	ax_ang1.axvline( color = 'k' )
	ax_act_pos1.axvline( color = 'k' )
	ax_act_ang1.axvline( color = 'k' )
	ax_pos2.axvline( color = 'k' )
	ax_ang2.axvline( color = 'k' )
	ax_act_pos2.axvline( color = 'k' )
	ax_act_ang2.axvline( color = 'k' )
	ax_cat_H.axhline( floor_z )

	ax_comp_time.plot( time_previous, [ 0. ] + mpc.times )
	ax_comp_time.axhline( sum( mpc.times ) / len( mpc.times ) )
	ax_nfeval.plot( time_previous, f_eval_record )
	ax_nfeval.axhline( sum( f_eval_record ) / len( f_eval_record ) )

	return figure


if __name__ == "__main__":

	n_frames = 300
	time_step = 0.01
	n_robots = 3
	state = zeros( (12 * n_robots,) )
	state[ 0 ] = 2
	state[ 6 ] = 2.5
	state[ 12 ] = 3
	actuation = zeros( (6 * (n_robots - 1),) )
	area = array( [ [ -4, 4 ], [ -4, 4 ], [ -2, 6 ] ] )
	max_actuation_x = 300.
	max_actuation_t = 1.
	floor_depth = 3.

	horizon = 25
	time_steps_per_act = 25

	key_frames = [ (0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 12),
								 (1., [ -3., 0., 0., 0., 0., 0. ] + [ 0. ] * 12),
								 (2., [ -3., 0., 0., 0., 0., 0. ] + [ 0. ] * 12) ]

	trajectory = generate_trajectory( key_frames, 2 * n_frames )
	trajectory[ :, 0, 2 ] = 1.3 * cos( 2.5 * (trajectory[ :, 0, 0 ] - 2) + pi ) + 1.3

	# plt.plot( diff( norm(trajectory[:, 0, :1], axis = 1) ) / time_step )
	# plt.show()
	# exit()

	model_kwargs = {
			"weight"                  : 11.5 * array( [ 0., 0., 9.81 ] ),
			"buoyancy"                : 120. * array( [ 0., 0., -1. ] ),
			"center_of_mass"          : array( [ 0., 0., 0. ] ),
			"center_of_volume"        : array( [ 0., 0., - 0.02 ] ),
			"inverted_inertial_matrix": inv(
					build_inertial_matrix( 11.5, array( [ 0., 0., 0. ] ), [ .16, .16, .16, 0.0, 0.0, 0.0 ] )
					),
			"hydrodynamic_matrix"     : diag( array( [ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07 ] ) )
			}

	pose_weight_matrix = eye( state.shape[ 0 ] // 2 )
	pose_weight_matrix[ 0:3, 0:3 ] *= 0.
	pose_weight_matrix[ 3:6, 3:6 ] *= 0.
	pose_weight_matrix[ 6:9, 6:9 ] *= 0.
	pose_weight_matrix[ 9:12, 9:12 ] *= 1.
	pose_weight_matrix[ 12:15, 12:15 ] *= 0.
	pose_weight_matrix[ 15:18, 15:18 ] *= 1.

	actuation_weight_matrix = eye( actuation.shape[ 0 ] )
	actuation_weight_matrix[ 0:3, 0:3 ] *= .01
	actuation_weight_matrix[ 3:6, 3:6 ] *= 1.
	actuation_weight_matrix[ 6:9, 6:9 ] *= .01
	actuation_weight_matrix[ 9:12, 9:12 ] *= 1.

	objective_weight = 100.
	final_cost_weight = 10.

	bluerov_chain = Model(
			three_robots_chain_follow, time_step, state, actuation, kwargs = model_kwargs, record = True
			)

	mpc_controller = MPC(
			bluerov_chain,
			horizon,
			trajectory,
			objective = chain_objective,
			time_steps_per_actuation = time_steps_per_act,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			objective_weight = objective_weight,
			final_weight = final_cost_weight,
			max_iter = 100,
			record = True
			)

	mpc_controller.constraints = (NonlinearConstraint(
			constraint_f,
			# -------H01---H12--dp01-dp12--dz01--------------dz12-------------vx1-vt1-vx2-vt2
			array(
					[ [ -inf, -inf, 0.4, 0.4, -2.6 / sqrt( 2 ), -2.6 / sqrt( 2 ), 0., 0., 0., 0. ] ]
					).repeat(
					horizon, axis = 0
					).flatten(),
			array(
					[ [ floor_depth, floor_depth, 2.6 / sqrt( 2 ), 2.6 / sqrt( 2 ), 2.6 / sqrt( 2 ),
							sqrt( 2 ) * 2.6, 5., .1, 5., .1 ] ]
					).repeat(
					horizon, axis = 0
					).flatten()
			),)

	previous_nfeval_record = [ 0 ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]

	logger = Logger()

	folder = (f'./plots/{three_robots_chain_follow.__name__}_'
						f'{int( time() )}')

	if path.exists( folder ):
		files_in_dir = glob( f'{folder}/*' )
		if len( files_in_dir ) > 0:
			if input( f"{folder} contains data. Remove? (y/n) " ) == 'y':
				for fig in files_in_dir:
					remove( fig )
			else:
				exit()
	else:
		mkdir( folder )

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( bluerov_chain.__dict__ | mpc_controller.__dict__, f, default = serialize_others )

	for frame in range( n_frames ):

		logger.log( f"frame {frame + 1}/{n_frames}" )

		mpc_controller.target_trajectory = trajectory[ frame + 1:frame + horizon + 1 ]

		bluerov_chain.state[ :6 ] = trajectory[ frame, 0, :6 ]
		# bluerov_chain.state[ 18:24 ] = diff(
		# 		trajectory[ frame:frame + 2, 0, :6 ], axis = 0
		# 		) / time_step

		mpc_controller.optimize()
		mpc_controller.apply_result()
		bluerov_chain.step()

		if not mpc_controller.raw_result.success:
			mpc_controller.tolerance = 1e-3
		else:
			mpc_controller.tolerance = 1e-6

		logger.log( f"{mpc_controller.times[ -1 ]:.3f}s" )
		logger.log( f"{mpc_controller.raw_result.success}" )
		logger.log( f"{mpc_controller.raw_result.nit}" )
		logger.log(
				f'{[ round( float( v ), 3 ) for v in constraint_f( mpc_controller.result ).reshape( (horizon, 10) )[ 0 ] ]}'
				)

		fig = plot(
				mpc_controller,
				trajectory,
				frame,
				n_frames,
				area,
				max_actuation_x,
				max_actuation_t,
				floor_depth,
				previous_nfeval_record,
				previous_H01_record,
				previous_H12_record
				)

		plt.savefig( f'{folder}/{frame}.png' )
		plt.close( 'all' )
		del fig

		logger.lognl( f'saved figure {frame}.png' )
		logger.save_at( folder )

	# create gif from frames
	logger.log( 'creating gif ...' )
	names = [ image for image in glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/animation.gif", append_images = frames, loop = True, save_all = True
			)
	logger.log( f'saved at {folder}/animation.gif' )
