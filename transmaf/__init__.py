from pkgutil import extend_path

from pkg_resources import DistributionNotFound, get_distribution

from .trainer import Trainer

__path__ = extend_path(__path__, __name__)  # type: ignore

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "0.0.0-unknown"
