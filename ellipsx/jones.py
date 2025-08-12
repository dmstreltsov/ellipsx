import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array


@jax.jit
def calc_chi_ambient(phi_deg: Float,
                     n0: Complex=1.0 + 0j) ->  tuple[Complex[Array, "2 2"], Complex[Array, "2 2"]]:
    """
    Calculate chi_0,pp and chi_0,ss characteristic matrices for the ambient
    (cf. (3.20a,b) in H. Tompkins, E. Irene, Handbook of ellipsometry, 2005)

    Args:
        phi_deg: incidence angle of the light beam in the ambient in degrees
        n0: refractive index of the ambient

    Returns:
        Tuple of chi_0,pp and chi_0,ss matrices   
    """
    phi = jnp.radians(phi_deg)
    cos_phi = jnp.cos(phi)
    chi_pp = 0.5*jnp.array([[1, cos_phi/n0],
                            [-1, cos_phi/n0]],
                           dtype=jnp.complex128)
    chi_ss = 0.5*jnp.array([[1, 1/(n0*cos_phi)],
                            [1, -1/(n0*cos_phi)]],
                          dtype=jnp.complex128)
    return (chi_pp, chi_ss)


@jax.jit
def calc_chi_substrate(phi_deg: Float,
                       n_sub: Complex,
                       n0: Complex=1.0 + 0j) -> tuple[Complex[Array, "2 2"], Complex[Array, "2 2"]]:
    """
    Calculate chi_sub,pp and chi_sub,ss characteristic matrices for the substrate
    (cf. (3.20c,d) in H. Tompkins, E. Irene, Handbook of ellipsometry, 2005)

    Args:
        phi_deg: incidence angle of the light beam in the ambient in degrees
        n_sub: complex refractive index of the substrate
        n0: refractive index of air

    Returns:
        Tuple of chi_sub,pp and chi_sub,ss matrices   
    """
    phi = jnp.radians(phi_deg)
    sin_phi = jnp.sin(phi)
    cos_phi_sub = jnp.sqrt(1 - jnp.power(sin_phi * n0 / n_sub, 2))
    chi_pp = jnp.array([[cos_phi_sub/n_sub, 0],
                        [1, 0]],
                       dtype=jnp.complex128)
    chi_ss = jnp.array([[1/(n_sub * cos_phi_sub), 0],
                        [1, 0]],
                       dtype=jnp.complex128)
    return (chi_pp, chi_ss)


@jax.jit
def calc_film_b(phi_deg: Float,
                n_f: Complex,
                d_f: Float,
                n0: Complex=1.0 + 0j,
                wl: Float=632.8) -> Complex:
    """
    Calculate b (film phase thickness, cf. (3.17b) in H. Tompkins, E. Irene, Handbook of ellipsomtery, 2005

    Args:
        phi_deg: incidence angle of the light beam in air in degrees
        n_f: complex refractive index of the film
        d_f: the film thickness in nm
        n0: refractive index of air
        wl: the incidence light wavelength in nm

    Retruns:
        b in radians
        
    """
    phi = jnp.radians(phi_deg)
    sin_phi = jnp.sin(phi)
    cos_phi_f = jnp.sqrt(1 - jnp.power(sin_phi*n0/n_f, 2))
    return 2*jnp.pi/wl * d_f * n_f * cos_phi_f


@jax.jit
def calc_p_matrices(phi_deg: Float,
                    n_f: Complex,
                    d_f: Float,
                    n0: Complex=1.0 + 0j,
                    wl: Float=632.8) -> tuple[Complex[Array, "2 2"], Complex[Array, "2 2"]]:
    """
    Calculate P_j,pp and P_j,ss matrices (the Abeles matrices)
    (cf. (3.18a,b) in H. Tompkins, E. Irene, Handbook of ellipsomtery, 2005)

    Args:
        phi_deg: incidence angle of the light beam in air in degrees
        n_f: complex refractive index of the film
        d_f: the film thickness in nm
        n0: refractive index of air
        wl: the incidence light wavelength in nm

    Returns:
        Tuple of P_j,pp and P_j,ss matrices
        
    """
    phi = jnp.radians(phi_deg)
    sin_phi = jnp.sin(phi)
    cos_phi_f = jnp.sqrt(1 - jnp.power(sin_phi*n0/n_f, 2))
    b_f = calc_film_b(phi_deg, n_f, d_f, n0, wl)
    p_pp= jnp.array([[jnp.cos(b_f), -1j * cos_phi_f / n_f * jnp.sin(b_f)],
                      [1j * n_f / cos_phi_f * jnp.sin(b_f), jnp.cos(b_f)]],
                    dtype=jnp.complex128)
    p_ss = jnp.array([[jnp.cos(b_f), 1j* jnp.sin(b_f) / (n_f * cos_phi_f)],
                      [1j * n_f * cos_phi_f * jnp.sin(b_f), jnp.cos(b_f)]],
                    dtype=jnp.complex128)
    return (p_pp, p_ss)








