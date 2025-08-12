import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Float


@jax.jit
def calc_delta_from_p1_p2(p1: tuple[Float, Float],
                          p2: tuple[Float, Float]) -> tuple[Float, Float]:
    """
    Compute the Delta ellipsometic angle value and its standard deviation from the values of
    polarizer azimuth angle measured in Zones 2 and 4.

    Args:
        p1: polarizer azimuth angle and its standard deviation measured in Zone 2 in degrees
        p2: polarizer azimuth angle and its standard deviation meausred in Zone 4 in degrees

    Returns:
        delta, delta_std: Delta ellipsometric angle and its standard deviation in degrees
    """
    
    delta = 360 - (p1[0] + p2[0])
    delta_std = jnp.sqrt(p1[1]**2 + p2[1]**2)

    return (jnp.where(delta > 360, delta - 360, delta),
            delta_std)
    

@jax.jit
def calc_psi_from_a1_a2(a1: tuple[Float, Float],
                        a2: tuple[Float, Float]) -> tuple[Float, Float]:
    """
    Compute the Psi ellipsometric angle value and its standard deviation from the values of
    analyzer azimuth angle measured in Zones 2 and 4.

    Args:
        a1: analyzer azimuth angle and its standard deviation measured in Zone 2 in degrees
        a2: analyzer azimuth angle and its standard deviation measured in Zone 4 in degrees

    Returns:
        psi, psi_std: Psi ellipsometic angle and its standard deviation in degrees
    """

    psi = 0.5 * (180 - (a2[0] - a1[0]))
    psi_std = 0.5 * jnp.sqrt(a1[1]**2 + a2[1]**2)
    return (psi, psi_std)
