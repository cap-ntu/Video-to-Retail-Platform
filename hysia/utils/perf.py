import os
import sys
import time
from typing import Callable


def timeit(func: Callable):
    """Use to record a running time of a method.

    This wrapper will record the running time (in ms) for invoking the wrapped function and log
    the time result. The result can be stored in a provided `dict` or printed.

    To store the result, we can provide an additional keyword argument `_log_time`, a `dict` data
    structure. This function will create or override an record with key name of the wrapped
    function full name or the user specified name.

    To specify the wrapped function name, we can provide an keyword argument `_log_name` when
    calling the wrapped function.

    Args:
        func (Callable): The wrapped function.

    Returns:
        The wrapped function called result.
    """

    def timed(*args, **kwargs):
        """Wrapper function of timing.
        """
        # obtain logger
        func_name = object_full_name(func)
        _log_time = kwargs.pop('_log_time', None)
        _log_name = kwargs.pop('_log_name', func_name)

        # timing
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()

        # log timing result
        if _log_time is not None:
            _log_time[_log_name] = (te - ts) * 1000
        else:
            print(f'{_log_name!r}  {(te - ts) * 1000:2.2f} ms')
        return result

    return timed


def object_full_name(obj: object):
    """Get a full qualified name of the object.

    This function will returned the full name, including the module name (class name or builtin)
    where the object is declared, and the object defined name. Each identical object will have
    different full name.

    From https://stackoverflow.com/a/2020083

    Args:
        obj (object): The object.

    Returns:
        str: The object's full qualified name.
    """
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + obj.__class__.__name__


class StreamSuppressor:
    """This is a wrapper for suppress print() on stdout.
    By https://stackoverflow.com/a/45669280

    FIXME: Not applicable to Python with C extension. Try this later:
        https://stackoverflow.com/a/17954769
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
