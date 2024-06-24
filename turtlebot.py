from glob import glob
from json import dump
from os import mkdir, path, remove
from time import perf_counter, time

from cycler import cycler
from matplotlib import pyplot as plt
from numpy import array, concatenate, cos, cumsum, eye, pi, sin
from PIL import Image

from mpc import *


# pendulum with cart
def turtle(
		x: ndarray, u: ndarray
		) -> ndarray:

	xdot = zeros( x.shape )
	_, _, theta, _, _, _ = x
	v, w = u
	xdot[ 0 ] = v * cos( theta )
	xdot[ 1 ] = v * sin( theta )
	xdot[ 2 ] = w

	return xdot


if __name__ == "__main__":

	state = array( [ 0., 0., 0., 0., 0., 0. ] )
	actuation = array( [ 0., 0. ] )

	model_kwargs = { }

	n_frames = 200
	max_iter = 1000
	tolerance = 1e-6
	time_step = 0.025

	sequence_time = n_frames * time_step

	base_optimization_horizon = 25
	optimization_horizon = base_optimization_horizon
	time_steps_per_actuation = 5

	optimization_horizon_lower_bound = 50

	pose_weight_matrix = eye( state.shape[ 0 ] // 2 )
	pose_weight_matrix[ 2, 2 ] = .1
	actuation_weight_matrix = .5 * eye( actuation.shape[ 0 ] )

	trajectory = [ (time_step * .2 * n_frames, [ 1., 1., pi ]),
								 (time_step * .4 * n_frames, [ -1., -1., -pi ]),
								 (time_step * .6 * n_frames, [ 1., 0, 0 ]),
								 (time_step * .8 * n_frames, [ 0, 0, pi ]),
								 (time_step * 1. * n_frames, [ 0, 1., 0 ]) ]

	command_upper_bound = 10
	command_lower_bound = -10

	mpc_config = {
			'candidate_shape'         : (
					optimization_horizon // time_steps_per_actuation + 1, actuation.shape[ 0 ]),

			'model'                   : turtle,
			'initial_actuation'       : actuation,
			'initial_state'           : state,
			'model_kwargs'            : model_kwargs,
			'target_pose'             : trajectory[ 0 ][ 1 ],
			'optimization_horizon'    : optimization_horizon,
			'prediction_horizon'      : 0,
			'time_step'               : time_step,
			'time_steps_per_actuation': time_steps_per_actuation,
			'objective_function'      : None,
			'pose_weight_matrix'      : pose_weight_matrix,
			'actuation_weight_matrix' : actuation_weight_matrix,
			'objective_weight'        : 0.,
			'final_cost_weight'       : 1.,
			'state_record'            : [ ],
			'actuation_record'        : [ ],
			'objective_record'        : None,
			'verbose'                 : False
			}

	result = zeros( mpc_config[ 'candidate_shape' ] )

	previous_states_record = [ deepcopy( state ) ]
	previous_actuation_record = [ deepcopy( actuation ) ]
	previous_target_record = [ ]

	note = ''

	folder = (f'./plots/{turtle.__name__}_'
						f'{note}_'
						f'dt={mpc_config[ "time_step" ]}_'
						f'opth={optimization_horizon}_'
						f'preh=_{mpc_config[ "prediction_horizon" ]}'
						f'dtpu={time_steps_per_actuation}_'
						f'{max_iter=}_'
						f'{tolerance=}_'
						f'{n_frames=}_'
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
		dump( mpc_config, f, default = serialize_others )

	for frame in range( n_frames ):

		if optimization_horizon > optimization_horizon_lower_bound:
			optimization_horizon -= 1

		for index in range( len( previous_target_record ) + 1, len( trajectory ) ):
			if trajectory[ index - 1 ][ 0 ] < frame * time_step:
				previous_target_record.append( trajectory[ index - 1 ] )
				mpc_config[ 'target_pose' ] = trajectory[ index ][ 1 ]
				optimization_horizon = base_optimization_horizon
				break

		mpc_config[ 'optimization_horizon' ] = optimization_horizon
		mpc_config[ 'candidate_shape' ] = (
				optimization_horizon // time_steps_per_actuation + 1, actuation.shape[ 0 ])

		mpc_config[ 'state_record' ] = [ ]
		mpc_config[ 'actuation_record' ] = [ ]

		print( f"frame {frame + 1}/{n_frames}\t", end = ' ' )

		result = result[ 1:mpc_config[ 'candidate_shape' ][ 0 ] ]
		difference = result.shape[ 0 ] - mpc_config[ 'candidate_shape' ][ 0 ]
		if difference < 0:
			result = concatenate( (result, array( [ [ 0., 0. ] ] * abs( difference ) )) )

		ti = perf_counter()

		result = optimize(
				cost_function = model_predictive_control_cost_function,
				cost_kwargs = mpc_config,
				initial_guess = result,
				tolerance = tolerance,
				max_iter = max_iter,
				constraints = NonlinearConstraint(
						lambda x: (actuation + cumsum(
								x.reshape( mpc_config[ 'candidate_shape' ] ), axis = 0
								)).flatten(), command_lower_bound, command_upper_bound
						)
				)

		actuation += result[ 0 ]
		state += turtle( state, actuation, **model_kwargs ) * time_step

		tf = perf_counter()
		compute_time = tf - ti

		mpc_config[ 'initial_state' ] = state
		mpc_config[ 'initial_actuation' ] = actuation

		previous_states_record.append( deepcopy( state ) )
		previous_actuation_record.append( deepcopy( actuation ) )

		n_f_eval = len( mpc_config[ 'state_record' ] )

		print(
				f"actuation={actuation}\t"
				f"state={state[ : state.shape[ 0 ] // 2 ]}\t"
				f"{compute_time=:.6f}s - {n_f_eval=}\t", end = ' '
				)

		ti = perf_counter()

		time_previous = [ i * time_step - (frame + 1) * time_step for i in range( frame + 2 ) ]
		time_prediction = [ i * time_step for i in range(
				mpc_config[ 'optimization_horizon' ] + mpc_config[ 'prediction_horizon' ]
				) ]

		fig = plt.figure()
		view = plt.subplot2grid( (3, 5), (0, 0), 4, 3, fig )
		view.grid( True )
		view.set_xlabel( "x" )
		view.set_ylabel( "y" )
		how = 11. / 8.9
		view.set_xlim( -2, 2 )
		view.set_ylim( -2 * how, 2 * how )

		ax_pos = plt.subplot2grid( (3, 5), (0, 3), 1, 2, fig )
		ax_pos.set_ylabel( 'position' )
		ax_pos.yaxis.set_label_position( "right" )
		ax_pos.yaxis.tick_right()
		ax_pos.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_pos.set_ylim( -3, 3 )
		ax_pos.set_prop_cycle( cycler( 'color', [ 'blue', 'red' ] ) )

		ax_ang = plt.subplot2grid( (3, 5), (1, 3), 1, 2, fig )
		ax_ang.set_ylabel( 'angle' )
		ax_ang.yaxis.set_label_position( "right" )
		ax_ang.yaxis.tick_right()
		ax_ang.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_ang.set_ylim( - pi, 2 * pi )
		ax_ang.set_prop_cycle( cycler( 'color', [ 'blue' ] ) )

		ax_act = plt.subplot2grid( (3, 5), (2, 3), 1, 2, fig )
		ax_act.set_ylabel( 'actuation' )
		ax_act.set_xlabel( 'time' )
		ax_act.yaxis.set_label_position( "right" )
		ax_act.yaxis.tick_right()
		ax_act.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
		ax_act.set_ylim( command_lower_bound, command_upper_bound )
		ax_act.set_prop_cycle( cycler( 'color', [ 'blue', 'red' ] ) )

		plt.subplots_adjust( hspace = 0., wspace = 0. )
		fig.suptitle( f"{frame + 1}/{n_frames} - {compute_time=:.6f}s - {n_f_eval=}" )

		view.scatter( state[ 0 ], state[ 1 ], c = 'r', s = 100 )
		view.quiver(
				state[ 0 ], state[ 1 ], .1 * cos( state[ 2 ] ), .1 * sin( state[ 2 ] ), color = 'b'
				)

		view.quiver(
				mpc_config[ 'target_pose' ][ 0 ],
				mpc_config[ 'target_pose' ][ 1 ],
				.1 * cos( mpc_config[ 'target_pose' ][ 2 ] ),
				.1 * sin( mpc_config[ 'target_pose' ][ 2 ] ),
				color = 'b'
				)

		t1 = 0.
		timespan = time_prediction[ -1 ] - time_previous[ 0 ]
		for index in range( len( previous_target_record ) ):
			t2 = (previous_target_record[ index ][ 0 ] - 2 * time_step) / timespan
			ax_pos.axhline(
					previous_target_record[ index ][ 1 ][ 0 ], t1, t2, color = 'b', linestyle = ':'
					)
			ax_pos.axhline(
					previous_target_record[ index ][ 1 ][ 1 ], t1, t2, color = 'r', linestyle = ':'
					)
			ax_ang.axhline(
					previous_target_record[ index ][ 1 ][ 2 ], t1, t2, color = 'b', linestyle = ':'
					)
			t1 = t2 + time_step

		ax_pos.axhline( mpc_config[ 'target_pose' ][ 0 ], t1, 1, color = 'b', linestyle = ':' )
		ax_pos.axhline( mpc_config[ 'target_pose' ][ 1 ], t1, 1, color = 'r', linestyle = ':' )
		ax_ang.axhline( mpc_config[ 'target_pose' ][ 2 ], t1, 1, color = 'b', linestyle = ':' )

		previous_pos_record_array = array( previous_states_record )[ :, :2 ]
		previous_ang_record_array = array( previous_states_record )[ :, 2 ]

		view.plot( previous_pos_record_array[ :, 0 ], previous_pos_record_array[ :, 1 ], 'b' )
		ax_pos.plot( time_previous, previous_pos_record_array )
		ax_ang.plot( time_previous, previous_ang_record_array )
		ax_act.plot( time_previous, previous_actuation_record )

		for f_eval in range( n_f_eval ):
			pos_record_array = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, :2 ]
			ang_record_array = array( mpc_config[ 'state_record' ][ f_eval ] )[ :, 2 ]

			view.plot( pos_record_array[ :, 0 ], pos_record_array[ :, 1 ], 'b', linewidth = .1 )

			ax_pos.plot( time_prediction, pos_record_array, linewidth = .1 )
			ax_ang.plot( time_prediction, ang_record_array, linewidth = .1 )
			ax_act.plot( time_prediction, mpc_config[ 'actuation_record' ][ f_eval ], linewidth = .1 )

		# plot vertical line from y min to y max
		ax_pos.axvline( color = 'k' )
		ax_ang.axvline( color = 'k' )
		ax_act.axvline( color = 'k' )

		plt.savefig( f'{folder}/{frame}.png' )
		plt.close( 'all' )
		del fig

		tf = perf_counter()
		save_time = tf - ti

		print( f'saved figure in {save_time:.6f}s\t', end = '' )
		print()

	# create gif from frames
	print( 'creating gif ...', end = ' ' )
	names = [ image for image in glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/animation.gif", append_images = frames, loop = True, save_all = True
			)
	print( f'saved at {folder}/animation.gif' )
