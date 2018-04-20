"""Defines the summary() function."""


def summary(obj, *args, **kwargs):
    """Summarize an object (supposed to mimic R's summary() function).

    Parameters
    ----------
    obj : object
        Object to summarize. If this object has a __summary__() method, this is
        called and returned. Otherwise, `obj` is printed together with its type
        and ID.
    args : sequence, optional
        Positional arguments to pass to __summary__().
    kwargs : sequence, optional
        Keyword arguments to pass to __summary__().

    Returns
    -------
    The return value of obj.__summary__(), if possible. Otherwise, obj itself is
    returned.
    """
    if hasattr(obj, "__summary__") and callable(obj.__summary__):
        return obj.__summary__(*args, **kwargs)
    else:
        print(obj)
        print(f"Type:  {obj.__class__.__name__}")
        print(f"ID:    0x{id(obj):x}")
        return obj
