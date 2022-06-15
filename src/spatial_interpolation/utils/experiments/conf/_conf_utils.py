def confmethod(func):
    """
    Decorator to flag a method as a configuration method.
    """
    func._confmethod = True   
    return func

def is_confmethod(func):
    """
    Check if a method is a configuration method.
    """
    return getattr(func, "_confmethod", False)