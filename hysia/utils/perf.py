import time

# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
    """a decorator for testing excution time like colde start
    
    Arguments:
        method {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{method.__name__!r}  {(te - ts) * 1000:2.2f} ms')
        return result
    return timed