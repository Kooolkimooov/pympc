from numpy import ndarray


class Dynamics:
    """
    abstract class for the dynamics of a system

    Methods
    -------
    **\_\_call\_\_**( *ndarray*, *ndarray*, *ndarray* ) -> *ndarray*:
        evaluates the dynamics
    **compute_error**( *ndarray*, *ndarray* ) -> *ndarray*:
        computes the error between two trajectories according to the system definition

    Properties
    ----------
    **state_size**: *int*:
        size of the expected state vector
    **actuation_size**: *int*:
        size of the expected actuation vector
    **position**: *ndarray*:
        indices of the position inside the state vector
    **orientation**: *ndarray*:
        indices of the orientation inside the state vector
    **velocity**: *ndarray*:
        indices of the velocity inside the state vector
    **body_rates**: *ndarray*:
        indices of the body rates inside the state vector
    **linear_actuation**: *ndarray*:
        indices of the linear actuation inside the actuation vector
    **angular_actuation**: *ndarray*:
        indices of the angular actuation inside the actuation vector
    """

    _state_size = None
    _actuation_size = None

    _position = None
    _orientation = None
    _velocity = None
    _body_rates = None

    _linear_actuation = None
    _angular_actuation = None

    def __call__( self, state: ndarray, actuation: ndarray, perturbation: ndarray ) -> ndarray:
        """
        evaluates the dynamics

        Parameters
        ----------
        state: ndarray
            current state of the system of shape (state_size,)
        actuation: ndarray
            current actuation of the system of shape (actuation_size,)
        perturbation: ndarray
            current perturbation of the system of shape (state_size / 2,)

        Returns
        -------
        ndarray: 
            state derivative of the system of shape (state_size,)
        """
        raise NotImplementedError

    def compute_error( self, actual: ndarray, target: ndarray ) -> ndarray:
        """
        computes the error between two trajectories of size (n, 1, state_size)

        Parameters
        ----------
        actual: ndarray
            actual state of the system of shape (n, 1, state_size)
        target: ndarray
            target state of the system of shape (n, 1, state_size)
        
        Returns
        -------
        ndarray: 
            error between the two trajectories of shape (n, 1, state_size)
        """
        raise NotImplementedError

    @property
    def state_size( self ) -> int:
        """size of the expected state vector (x, x_dot)"""
        return self._state_size

    @property
    def actuation_size( self ) -> int:
        """size of the expected actuation vector"""
        return self._actuation_size

    @property
    def position( self ) -> ndarray:
        """indices of the position inside the state vector"""
        return self._position

    @property
    def orientation( self ) -> ndarray:
        """indices of the orientation inside the state vector"""
        return self._orientation

    @property
    def velocity( self ) -> ndarray:
        """indices of the velocity inside the state vector"""
        return self._velocity

    @property
    def body_rates( self ) -> ndarray:
        """indices of the body rates inside the state vector"""
        return self._body_rates

    @property
    def linear_actuation( self ) -> ndarray:
        """indices of the linear actuation inside the actuation vector"""
        return self._linear_actuation

    @property
    def angular_actuation( self ) -> ndarray:
        """indices of the angular actuation inside the actuation vector"""
        return self._angular_actuation
