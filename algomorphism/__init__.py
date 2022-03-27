try:
    from . import base, layers, losses, models, metrics, methods, figures
except ImportError as ie:
    pass

__version__ = '1.0.0-beta'
