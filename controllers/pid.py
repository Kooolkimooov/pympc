from time import perf_counter

from numpy import eye, ndarray, zeros

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
            record: bool = False,
            verbose: bool = False
    ):
        self.model = model
        self.gain_shape = (self.model.dynamics.actuation_size, self.model.dynamics.state_size // 2)
        self.result_shape = (self.model.dynamics.actuation_size,)

        self.target = target

        self.last_error = None
        self.integral_error = zeros( (1, 1, self.model.dynamics.state_size // 2) )

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
            self.offset = zeros( self.result_shape )
        else:
            assert offset.shape == self.result_shape, f'offset gain must be of shape {self.result_shape}'
            self.offset = derivative

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

        error = self.model.dynamics.compute_error(
                self.model.state[ :self.model.dynamics.state_size // 2 ].reshape(
                        (1, 1, self.model.dynamics.state_size // 2)
                ),
                self.target.reshape( (1, 1, self.model.dynamics.state_size // 2) )
        )

        self.integral_error += error * self.model.time_step
        derivative_error = zeros( (1, 1, self.model.dynamics.state_size // 2) )

        if not self.last_error is None:
            derivative_error = (error - self.last_error) / self.model.time_step

        self.last_error = error

        world_to_body = self.model.dynamics.get_body_to_world_transform( self.model.state ).T

        actuation = zeros( self.result_shape )
        actuation += self.proportional @ world_to_body @ error.flatten()
        actuation += self.integral @ world_to_body @ self.integral_error.flatten()
        actuation += self.derivative @ world_to_body @ derivative_error.flatten()
        actuation += self.offset

        if self.record:
            self.compute_times.append( perf_counter() - ti )

        if self.verbose:
            print( f'Error    : {error.flatten()}' )
            print( f'P        : {(self.proportional @ error.flatten())}' )
            print( f'I        : {(self.integral @ self.integral_error.flatten())}' )
            print( f'D        : {(self.derivative @ derivative_error.flatten())}' )
            print( f'Actuation: {actuation}' )
            print( f'Time     : {self.compute_times}' )

        return actuation
