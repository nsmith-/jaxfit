"""JaxFit

Some interesting text
"""

from ._version import version as __version__

from jax.config import config as _config

# TODO: fine-grain control
# Really the only place float32 falls apart is poisson logpmf for very large N
_config.update("jax_enable_x64", True)

from ._version import version as __version__
