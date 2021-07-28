"""JaxFit

Some interesting text
"""
from jax.config import config as _config

# TODO: fine-grain control
# Really the only place float32 falls apart is poisson logpmf for very large N
_config.update("jax_enable_x64", True)
