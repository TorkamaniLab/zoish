import functools


def trackcalls(func):
    """Helper function to check if
    a method already called or not
    Parameters
    ----------
    func : A function or method
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)

    wrapper.has_been_called = False
    return wrapper
