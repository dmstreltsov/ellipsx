"""
Effective medium approximations
"""
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Complex, Float


@partial(jax.jit, static_argnames=('f',))
def bruggeman(n1: Complex,
              n2: Complex,
              f: Float = 0.5) -> Complex:
    """
    Effective medium approximation for two media using the Bruggeman model

    Args:
        n1: complex refractive index of the first medium
        n2: complex refractive index of the second medium
        f:  fraction of the first medium

    Returns:
        Complex refractive index of the effective medium
    """
    if f < 0:
        raise ValueError("Fraction should be non negative")
    if f > 1:
        raise ValueError("Fraction should not be greater that 1")

    hb = (3 * f - 1) * n1**2 + (3 * (1 - f) - 1) * n2**2
    n_eff = 0.5*jnp.sqrt(hb + jnp.sqrt(hb**2 + 8 * n1**2 * n2**2))
    return n_eff
