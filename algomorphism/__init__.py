# author: Efthymis Michalis

try:
    from . import method, figure, model, dataset
except ImportError as ie:
    pass

__version__ = '1.0.0'
