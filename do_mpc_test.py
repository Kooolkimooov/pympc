import importlib.util
import time

import numpy as np
import sys
from casadi import *

# Import do_mpc package:
import do_mpc
from matplotlib import pyplot as plt

def get_state_matrixes( state, robot_configuration = None ):
	if robot_configuration is None:
		robot_configuration = bluerov_configuration
	mass = robot_configuration[ "mass" ]
	center_of_mass = robot_configuration[ "center_of_mass" ]
	Iinv = robot_configuration["inertial_matrix_inv"]

	Phi, Theta, Psi = state[ 3 ], state[ 4 ], state[ 5 ]
	cPhi, sPhi = np.cos( Phi ), np.sin( Phi )
	cTheta, sTheta, tTheta = np.cos( Theta ), np.sin( Theta ), np.tan( Theta )
	cPsi, sPsi = np.cos( Psi ), np.sin( Psi )
	J = casadi.SX.zeros( (6, 6) )
	J[ 0, :3 ] = np.array(
			[ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi, sPsi * sPhi + cPsi * sTheta * cPhi ]
			)
	J[ 1, :3 ] = np.array(
			[ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi, -cPsi * sPhi + sPsi * sTheta * cPhi ]
			)
	J[ 2, :3 ] = np.array( [ -sTheta, cTheta * sPhi, cTheta * cPhi ] )
	J[ 3, 3: ] = np.array( [ 1, sPhi * tTheta, cPhi * tTheta ] )
	J[ 4, 3: ] = np.array( [ 0, cPhi, -sPhi ] )
	J[ 5, 3: ] = np.array( [ 0, sPhi / cTheta, cPhi / cTheta ] )

	hydrodynamic_coefficients = robot_configuration[ "hydrodynamic_coefficients" ]
	D = np.multiply( np.eye( 6 ), hydrodynamic_coefficients[ :6 ] ) + np.multiply(
			np.eye( 6 ), hydrodynamic_coefficients[ 6: ] * norm_2( state[ 6: ] )
			)
	buoyancy = robot_configuration[ "buoyancy" ]
	center_of_volume = robot_configuration[ "center_of_volume" ]
	Fw = mass * np.array( [ 0, 0, 9.80665 ] )
	Fb = buoyancy * np.array( [ 0, 0, 1 ] )
	S = casadi.SX( np.zeros( 6 ) )
	S[ :3 ] = J[ :3, :3 ].T @ (Fw + Fb)
	S[ 3: ] = J[ :3, :3 ].T @ (np.cross( center_of_mass, Fw ) + np.cross( center_of_volume, Fb ))

	return J, Iinv, D, S

if __name__ == '__main__':
	ti = time.perf_counter()

	model_type = 'continuous'  # either 'discrete' or 'continuous'
	model = do_mpc.model.Model( model_type )

	mass = 11.5
	inertial_coefficients = np.array( [ .16, .16, .16, 0.0, 0.0, 0.0 ] )
	center_of_mass = np.array( [ 0.0, 0.0, 0.0 ] )
	inertial_matrix = np.eye( 6 )
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

	bluerov_configuration = {
			"mass"                     : mass,
			"center_of_mass"           : center_of_mass,
			"buoyancy"                 : 120.0,
			"center_of_volume"         : np.array( [ 0.0, 0.0, - 0.02 ] ),
			"inertial_matrix_inv"      : np.linalg.inv( inertial_matrix ),
			"hydrodynamic_coefficients": np.array(
					[ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
					),
			"robot_max_actuation"      : np.array( [ 1000, 1000, 1000, 1000, 1000, 1000 ] ),
			"robot_max_actuation_ramp" : np.array( [ 100, 100, 100, 100, 100, 100 ] )
			}

	horizon = 4
	r_horizon = 4
	nsteps = 25
	dt = .5
	target = casadi.SX( np.array( [ .5, -.5, 1, 0, 0, pi ] ) )

	setup_mpc = {
			'n_horizon': horizon, 'n_robust': r_horizon, 'open_loop': False, 't_step': dt,
			# 'state_discretization': 'collocation',
			# 'collocation_type'    : 'radau',
			# 'collocation_deg'     : 5,
			# 'collocation_ni'      : 5,
			# 'store_full_solution' : True,

			# Use MA27 linear solver in ipopt for faster calculations:
			# 'nlpsol_opts'         : { 'ipopt.linear_solver': 'spral'} # 'mumps' } #
			}

	params_simulator = {
			# Note: cvode doesn't support DAE systems.
			'integration_tool': 'idas', # 'abstol': 1e-9,
			# 'reltol': 1e-9,
			't_step'          : dt
			}

	eta = model.set_variable( '_x', 'eta', shape = (6, 1) )
	deta = model.set_variable( '_x', 'deta', shape = (6, 1) )
	nu = model.set_variable( '_x', 'nu', shape = (6, 1) )
	dnu = model.set_variable( '_x', 'dnu', shape = (6, 1) )

	u = model.set_variable( '_u', 'force', shape = (6, 1) )

	cost = model.set_expression(
			'model_predictive_control_cost_function', sum1( (target - eta) ** 2 )
			)

	J, Iinv, D, S = get_state_matrixes( eta )

	model.set_rhs( 'deta', J @ nu )
	model.set_rhs( 'dnu', Iinv @ (D @ nu + S + u) )
	model.set_rhs( 'eta', deta )
	model.set_rhs( 'nu', dnu )

	model.setup()

	mpc = do_mpc.controller.MPC( model )
	mpc.set_param( **setup_mpc )
	mpc.settings.supress_ipopt_output()

	mterm = model.aux[ 'model_predictive_control_cost_function' ]
	lterm = model.aux[ 'model_predictive_control_cost_function' ]

	mpc.set_objective( mterm = mterm, lterm = lterm )
	mpc.set_rterm( force = 0 )

	# bounds on force
	mpc.bounds[ 'lower', '_u', 'force' ] = - bluerov_configuration[ "robot_max_actuation" ]
	mpc.bounds[ 'upper', '_u', 'force' ] = bluerov_configuration[ "robot_max_actuation" ]

	mpc.prepare_nlp()
	# bounds on force derivative
	for i in range( horizon - 1 ):
		for j in range( 6 ):
			mpc.nlp_cons.append( (mpc.opt_x[ '_u', i + 1, 0 ][ j ] - mpc.opt_x[ '_u', i, 0 ][ j ]) )
			mpc.nlp_cons_lb.append( - bluerov_configuration[ 'robot_max_actuation_ramp' ][ j ] * dt )
			mpc.nlp_cons_ub.append( bluerov_configuration[ 'robot_max_actuation_ramp' ][ j ] * dt )
	mpc.setup()

	estimator = do_mpc.estimator.StateFeedback( model )
	simulator = do_mpc.simulator.Simulator( model )
	simulator.set_param( **params_simulator )
	simulator.setup()

	simulator.x0[ 'eta' ] = np.zeros( 6 )
	simulator.x0[ 'nu' ] = np.zeros( 6 )

	x0 = simulator.x0
	mpc.x0 = x0
	mpc.u0 = np.zeros( 6 )
	estimator.x0 = x0
	mpc.set_initial_guess()

	sim_graphics = do_mpc.graphics.Graphics( simulator.data )
	fig, ax = plt.subplots( 3, sharex = True, figsize = (16, 9) )
	fig.align_ylabels()
	sim_graphics.add_line( var_type = '_x', var_name = 'eta', axis = ax[ 0 ] )
	sim_graphics.add_line( var_type = '_u', var_name = 'force', axis = ax[ 1 ] )

	computation_times = [ ]

	tf = time.perf_counter()
	print( f'initialization in {tf - ti = }' )

	for i in range( 1, nsteps + 1 ):
		print( f' -> {dt = } s | {horizon = } | {i = } / {nsteps} | {i * dt = :.3f} s | {target = }' )
		ti = time.perf_counter()
		u0 = mpc.make_step( x0 )
		y0 = simulator.make_step( u0 )
		x0 = estimator.make_step( y0 )
		tf = time.perf_counter()
		computation_times += [ tf - ti ]
		print(
				f'\tdt = {tf - ti:.3f} s | '
				f'median = {np.median( computation_times ):.3f} s | '
				f'min = {min( computation_times ):.3f} s | '
				f'1% = {np.percentile( computation_times, 1 ):.3f} s | '
				f'25% = {np.percentile( computation_times, 25 ):.3f} s | '
				f'75% = {np.percentile( computation_times, 75 ):.3f} s | '
				f'99% = {np.percentile( computation_times, 99 ):.3f} s | '
				f'max = {max( computation_times ):.3f} s | '
				f'total time = {sum( computation_times ):.3f} s'
				)
		print( '\tu =', str( u0 ).replace( '\n', '' ) )
		print( '\tx =', str( x0[ :6 ] ).replace( '\n', '' ) )

	ax[ 2 ].plot( [ k * dt for k in range( nsteps ) ], computation_times )
	ax[ 2 ].axhline( dt )
	sim_graphics.plot_results()
	sim_graphics.reset_axes()
	plt.show()
