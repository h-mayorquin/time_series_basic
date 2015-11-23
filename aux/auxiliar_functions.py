"""
Auxiliar functions that are not thematically unified
"""

def sizeof_fmt(num, suffix='B'):
    """
    This is a function that returns the size of a function in a human readable way
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
