from numpy import eye, ndarray, zeros

from pympc.models.model import Model


class PID:
    """

    """

    def __init__(
            self,
            model: Model,
            target: ndarray,
            proportional: ndarray = None,
            integral: ndarray = None,
            derivative: ndarray = None,
            record: bool = False,
            verbose: bool = False
    ):
        self.model = model
        self.gain_shape = (self.model.dynamics.actuation_size, self.model.dynamics.state_size // 2)

        self.target = target

        self.last_error = None
        self.integral_error = zeros( (1, 1, self.model.dynamics.state_size) )

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

        self.verbose = verbose
        self.record = record

    def step( self ) -> ndarray:
        """
        computes the actuation with target and model state. records the computation
        time if record is True and returns the best actuation

        Returns
        -------
        ndarray:
            actuation shape = (actuation_size,)
        """
        error = self.model.dynamics.compute_error(
                self.model.state.reshape( (1, 1, self.model.dynamics.state_size) ),
                self.target.reshape( (1, 1, self.model.dynamics.state_size) )
        )

        self.integral_error += error * self.model.time_step
        derivative_error = zeros( (1, 1, self.model.dynamics.state_size) )

        if not self.last_error is None:
            derivative_error = (error - self.last_error) / self.model.time_step

        self.last_error = error

        actuation = zeros( (self.model.dynamics.actuation_size,) )
        actuation += self.proportional @ error.flatten()
        actuation += self.integral @ self.integral_error.flatten()
        actuation += self.derivative @ derivative_error.flatten()

        return actuation
