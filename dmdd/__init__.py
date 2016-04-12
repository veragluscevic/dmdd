__version__ = '0.2'

try:
    __DMDD_SETUP__
except NameError:
    __DMDD_SETUP__ = False

if not __DMDD_SETUP__:
    from .dmdd import *

