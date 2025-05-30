from glob import glob
from inspect import getsource, isfunction, ismethod
from os import mkdir, path, remove
from platform import release, system, version

from PIL import Image
from cpuinfo import get_cpu_info
from numpy import array, ndarray, zeros

G: float = 9.80665
rho_eau: float = 997.


class Logger:
    def __init__( self, print_to_terminal: bool = True ):
        self.logs: str = ''
        self.print_to_terminal = print_to_terminal

    def log( self, log: str ):
        """
        :param log: text to be printed and saved. ends with a tabulation
        :return: None
        """
        self.logs += log
        self.logs += '\t'
        if self.print_to_terminal:
            print( log, end='\t' )

    def lognl( self, log: str ):
        """
        :param log: text to be printed and saved. ends with a new line
        :return: None
        """
        self.logs += log
        self.logs += '\n'
        if self.print_to_terminal:
            print( log )

    def logrl( self, log: str ):
        """
        :param log: text to be printed and saved. ends with a return to the beginning of the line,
        the saved text goes to a new line
        :return: None
        """
        self.logs += log
        self.logs += '\n'
        if self.print_to_terminal:
            print( log, end='\r' )

    def save_at( self, path: str, file: str = 'logs' ):
        """
        :param path: folder in which to save the current log
        :param file: name of the file
        """
        with open( f'{path}/logs.txt', 'w' ) as f:
            f.write( self.logs )


def generate_trajectory(
        key_frames: list, n_points: int
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


def check( folder: str, recursive=False, prompt=True ) -> int:
    """
    counts the number of objects in a folder and removes them if the user agrees
    if the folder does not exist, it creates it
    :param folder:
    :param recursive:
    :param prompt:
    :return: number of objects in the folder
    """
    n = 0
    if path.exists( folder ):
        objects_in_dir = glob( f'{folder}/*' )
        n += len( objects_in_dir )
        if n > 0:
            if prompt and (input( f"{folder} exists and contains data. Remove? (y/n) " ) == 'y'):
                for object in objects_in_dir:
                    if path.isdir( object ) and recursive:
                        n += check( object )
                    else:
                        print( f'removing {object}', end='\t' )
                        try:
                            remove( object )
                            print( 'success.' )
                        except:
                            print( 'failed ...' )
                        n -= 1
    else:
        mkdir( folder )

    return n


def gif_from_pngs( folder: str, duration: float = None ):
    if duration is None:
        duration = 33.
    names = [ image for image in glob( f"{folder}/*.png" ) ]
    names.sort( key=lambda x: path.getmtime( x ) )
    frames = [ Image.open( name ) for name in names ]
    frame_one = frames[ 0 ]
    frame_one.save(
            f"{folder}/animation.gif", append_images=frames, loop=True, save_all=True, duration=duration
    )


def serialize_others( obj: any ):
    if isfunction( obj ) or ismethod( obj ) or isinstance( obj, staticmethod ):
        return getsource( obj )
    if isinstance( obj, ndarray ):
        return obj.tolist()
    if isinstance( obj, slice ):
        return f'{obj.start}:{obj.stop}:{obj.step}'
    if isinstance( obj, bool ):
        return str( obj )
    try:
        output = { }
        # class attribute priority is for left most class in inheritance list,
        # we reverse the __bases__ list to get the correct order
        for base in reversed( get_all_bases( obj.__class__ ) ):
            output |= base.__dict__
        output |= obj.__dict__
        output |= obj.__class__.__dict__
        output[ 'instance_of' ] = obj.__class__.__name__

        return output
    except:
        pass


def get_all_bases( obj: any ):
    bases = tuple()
    for base in obj.__bases__:
        if base.__name__ == 'object':
            continue
        bases += (base,)
        bases += get_all_bases( base )
    return bases


def print_dict( d: dict, max_list_size: int = 10, prefix: str = '' ):
    for k, v in d.items():
        if k[ 0 ] == '_':
            continue

        if isinstance( v, dict ):
            print( prefix + k, type( v ), ':', flush=True )
            print_dict( v, max_list_size, prefix + '\t' )
            continue

        if isinstance( v, list ):
            if len( v ) > 0 and isinstance( v[ 0 ], dict ):
                print( prefix + k, type( v ), ':', flush=True )
                print_dict( { str( i ): e for i, e in enumerate( v ) }, max_list_size, prefix + '\t' )
                continue

            l = array( v ).shape
            print( prefix + k, type( v ), ':', v if sum( l ) < max_list_size else l, flush=True )
            continue

        if isinstance( v, str ) and len( v ) > max_list_size:
            print( prefix + k, type( v ), ':', v[ :max_list_size ], '...', flush=True )
            continue

        print( prefix + k, type( v ), ':', v, flush=True )


def compare_dict( d0: dict, d1: dict, max_list_size: int = 10, prefix: str = '', only_diff=False ):
    keys = set( list( d0.keys() ) + list( d1.keys() ) )

    for k in keys:
        if k[ 0 ] == '_':
            continue

        v0 = d0.get( k )
        v1 = d1.get( k )

        equal = v0 == v1

        if only_diff and equal:
            continue

        v = v0 if v0 is not None else v1

        if v0 is None:
            v0 = v1.__class__()
        if v1 is None:
            v1 = v0.__class__()

        color_prefix = '\033[1;102m' if equal else '\033[1;101m'
        color_suffix = '\033[0m'

        if type( v0 ) != type( v1 ):
            print( prefix + color_prefix + k, type( v0 ), '!=', type( v1 ), color_suffix, flush=True )
            continue

        if isinstance( v, dict ):
            print( prefix + color_prefix + k, type( v ), color_suffix + ':', flush=True )
            compare_dict(
                    d0=v0, d1=v1, max_list_size=max_list_size, prefix=prefix + '\t', only_diff=only_diff
            )
            continue

        if isinstance( v, list ):
            if len( v ) > 0 and isinstance( v[ 0 ], dict ):
                print( prefix + color_prefix + k, type( v ), color_suffix + ':', flush=True )
                compare_dict(
                        d0={ str( i ): e for i, e in enumerate( v0 ) },
                        d1={ str( i ): e for i, e in enumerate( v1 ) },
                        max_list_size=max_list_size,
                        prefix=prefix + '\t',
                        only_diff=only_diff
                )
                continue

            l = array( v ).shape
            if sum( l ) < max_list_size:
                print(
                        prefix + color_prefix + k, type( v ), color_suffix + ':', v0, v1, flush=True
                )
            else:
                print(
                        prefix + color_prefix + k, type( v ), color_suffix + ':', len( v0 ), len( v1 ), flush=True
                )
            continue

        if isinstance( v, str ) and len( v ) > max_list_size:
            print(
                    prefix + color_prefix + k,
                    type( v ),
                    color_suffix + ':',
                    v0[ :max_list_size ],
                    v1[ :max_list_size ],
                    flush=True
            )
            continue

        print(
                prefix + color_prefix + k, type( v ), color_suffix + ':', v0, v1, flush=True
        )


def get_computer_info():
    info = { }
    info[ 'os' ] = system()
    info[ 'os_release' ] = release()
    info[ 'os_version' ] = version()
    info[ 'cpu' ] = get_cpu_info()

    return info


def runge_kutta_4(
        function: callable, time_step: float, current_state: ndarray, *args, **kwargs
):
    """
    Runge-Kutta 4th order method
    :param function: function to integrate, must have the following signature: f(x, *args, **kwargs)
    :param time_step: time step
    :param current_state: initial position
    :param args: additional arguments for f
    :param kwargs: additional keyword arguments for f
    """

    # coefficients of the Butcher tableau
    a21 = .4
    a31 = .29697761
    a32 = .15875964
    a41 = .21810040
    a42 = -3.05096516
    a43 = 3.83286476
    b1 = .17476028
    b2 = -.55148066
    b3 = 1.20553560
    b4 = .17118478

    k1 = function( current_state, *args, **kwargs )
    k2 = function( current_state + a21 * k1 * time_step, *args, **kwargs )
    k3 = function( current_state + (a31 * k1 + a32 * k2) * time_step, *args, **kwargs )
    k4 = function( current_state + (a41 * k1 + a42 * k2 + a43 * k3) * time_step, *args, **kwargs )

    new_state = current_state + (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4) * time_step

    return new_state
