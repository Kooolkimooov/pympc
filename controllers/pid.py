from time import perf_counter

from numpy import abs, any, eye, ndarray, zeros

from pympc.models.model import Model


class PID:
    """
    Proportional–Integral–Derivative (PID) controller for a given model.

    Parameters
    ----------
    model : Model
        The dynamical model to be controlled. Must provide:
    target : ndarray
        Desired target state. Expected shape `(state_size,)`.
    proportional : ndarray, optional
        Proportional gain matrix. If omitted, defaults to an identity
    integral : ndarray, optional
        Integral gain matrix. If omitted, defaults to an identity
    derivative : ndarray, optional
        Derivative gain matrix. If omitted, defaults to an identity
    offset : ndarray, optional
        offset vector. If omitted, defaults to zeros
    record : bool, default False
        Whether to record timing or diagnostics.
    verbose : bool, default False
        Whether to print debugging information.

    Methods
    -------
    - step() -> ndarray
        Compute actuation for the current state and target using the PID law.

    Properties
    ----------
    - model : Model
        Controlled model instance.
    - target : ndarray
        Target state; shape `(state_size,)`.
    - gain_shape : tuple
        `(actuation_size, state_size // 2)` against which custom gains are validated.
    - proportional : ndarray
        Gain matrices used in the PID law.
    - integral : ndarray
        Gain matrices used in the PID law.
    - derivative : ndarray
        Gain matrices used in the PID law.
    - offset : ndarray
        offset used in the PID law.
    - last_error : ndarray | None
        Last error used for the derivative term; shape `(1, 1, state_size)` when set.
    - integral_error : ndarray
        Accumulated integral error; shape `(1, 1, state_size)`.
    - record, verbose : bool
        Configuration flags.
    """

    def __init__(
            self,
            model: Model,
            target: ndarray,
            proportional: ndarray = None,
            integral: ndarray = None,
            derivative: ndarray = None,
            offset: ndarray = None,
            use_anti_windup: bool = True,
            anti_windup_limit: float = 1e1,
            record: bool = False,
            verbose: bool = False
    ):
        self.model = model
        self.gain_shape = (self.model.dynamics.actuation_size, self.model.dynamics.state_size // 2)
        self.result_shape = (self.model.dynamics.actuation_size,)

        self.target = target

        self.error = zeros( (self.model.dynamics.state_size // 2,) )
        self.integral_error = zeros( (self.model.dynamics.state_size // 2,) )
        self.derivative_error = zeros( (self.model.dynamics.state_size // 2,) )

        if proportional is None:
            self.proportional = eye( self.model.dynamics.state_size )
            self.proportional = self.proportional[ self.model.dynamics.six_dof_actuation_mask, : ]
        else:
            assert proportional.shape == self.gain_shape, f'proportional gain must be of shape {self.gain_shape}'
            self.proportional = proportional

        if integral is None:
            self.integral = eye( self.model.dynamics.state_size )
            self.integral = self.integral[ self.model.dynamics.six_dof_actuation_mask, : ]
        else:
            assert integral.shape == self.gain_shape, f'integral gain must be of shape {self.gain_shape}'
            self.integral = integral

        if derivative is None:
            self.derivative = eye( self.model.dynamics.state_size )
            self.derivative = self.derivative[ self.model.dynamics.six_dof_actuation_mask, : ]
        else:
            assert derivative.shape == self.gain_shape, f'derivative gain must be of shape {self.gain_shape}'
            self.derivative = derivative

        if offset is None:
            self.offset = zeros( (self.model.dynamics.actuation_size,) )
        else:
            assert offset.shape == self.result_shape, f'offset gain must be of shape {self.result_shape}'
            self.offset = offset

        self.use_anti_windup = use_anti_windup

        assert anti_windup_limit > 0, 'anti_windup_limit must be positive'
        self.anti_windup_limit = anti_windup_limit

        self.verbose = verbose
        self.record = record

        self.compute_times = [ ]

    def step( self ) -> ndarray:
        """
        Compute the actuation for the current model state toward the target.

        The error is computed via `model.dynamics.compute_error` on full-state tensors
        of shape `(1, 1, state_size)`. The integral and derivative terms use
        `model.time_step` as the sampling time.

        Returns
        -------
        ndarray
            Actuation vector with shape `(actuation_size,)`.
        """

        if self.record:
            ti = perf_counter()

        body_to_world = self.model.dynamics.get_body_to_world_transform( self.model.state )

        self.error = self.model.dynamics.compute_error(
                self.model.state[ :self.model.dynamics.state_size // 2 ].reshape(
                        (1, 1, self.model.dynamics.state_size // 2)
                ),
                self.target.reshape( (1, 1, self.model.dynamics.state_size // 2) )
        ).flatten()

        if self.use_anti_windup and any( abs( self.integral_error ) > self.anti_windup_limit ):
            self.integral_error.fill( 0.0 )
        self.integral_error += self.error * self.model.time_step

        self.derivative_error = zeros( (self.model.dynamics.state_size // 2, ) )

        self.derivative_error = - body_to_world @ self.model.state[ self.model.dynamics.state_size // 2: ]

        actuation = zeros( self.result_shape )
        actuation += self.proportional @ body_to_world.T @ self.error.flatten()
        actuation += self.integral @ body_to_world.T @ self.integral_error.flatten()
        actuation += self.derivative @ body_to_world.T @ self.derivative_error.flatten()
        actuation += self.offset

        if self.record:
            self.compute_times.append( perf_counter() - ti )

        if self.verbose:
            print( f'Error    : {self.error.flatten()}' )
            print( f'P        : {self.proportional @ body_to_world.T @ self.error.flatten()}' )
            print( f'I        : {self.integral @ body_to_world.T @ self.integral_error.flatten()}' )
            print( f'D        : {self.derivative @ body_to_world.T @ self.derivative_error.flatten()}' )
            print( f'Actuation: {actuation}' )
            print( f'Time     : {self.compute_times}' )

        return actuation
