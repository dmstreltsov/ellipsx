"""
Models of the samples
"""
import jax
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array

from .jones import calc_chi_substrate, calc_p_matrices, calc_chi_ambient
from .ema import bruggeman


@jax.jit
def calc_m_matrices_one_layer(phi_deg: Float,
                              n_f: Complex,
                              d_f: Float,
                              n_sub: Complex,
                              n0: Complex = 1.0 + 0j,
                              wl: Float = 632.8) -> tuple[Complex[Array, "2 2"],
                                                          Complex[Array, "2 2"]]:
    """
    Calculate the characteristic matrices M_pp and M_ss for the layer stack consisting of
    one uniform film on the substrate
    (cf. (3.19a,b) in H. Tompkins, E. Irene, Handbook of ellipsomtery, 2005)

    Args:
        phi_deg: incidence angle of the light beam in air in degrees
        n_f: complex refractive index of the film
        d_f: the film thickness in nm
        n_sub: complex refractive index of the substrate
        n0: refractive index of air
        wl: the incidence light wavelength in nm

    Returns:
        Tuple of M_pp and M_ss matrices
    """
    chi_sub_pp, chi_sub_ss = calc_chi_substrate(phi_deg, n_sub, n0)
    p_pp, p_ss = calc_p_matrices(phi_deg, n_f, d_f, n0, wl)
    chi_0_pp, chi_0_ss = calc_chi_ambient(phi_deg, n0)
    m_pp = chi_0_pp @ p_pp @ chi_sub_pp
    m_ss = chi_0_ss @ p_ss @ chi_sub_ss
    return (m_pp, m_ss)


@jax.jit
def calc_m_matrices_three_layers(phi_deg: Float,
                                 n_f: Complex,
                                 d_f: Float,
                                 n_sub: Complex,
                                 d_0f: Float,
                                 d_fs: Float,
                                 n0: Complex = 1.0 + 0j,
                                 wl: Float = 632.8) -> tuple[Complex[Array, "2 2"],
                                                             Complex[Array, "2 2"]]:
    """
    Calculate the characteristic matrices M_pp and M_ss for the layer stack consisting of
    three layers on the substrate, i.e. the uniform film layer, the interface layer between
    the film and the ambient, and the  interface layer between the film and the substrate.
    The refractive index of the interface layer is modelled
    with the Bruggeman EMA, assuming equal content of both adjacent phases.

    Args:
        phi_deg: incidence angle of the light beam in air in degrees
        n_f: complex refractive index of the film
        d_f: the film thickness in nm
        n_sub: complex refractive index of the substrate
        d_0f: interface width between the film and the ambient in nm
        d_fs: interface width between the film and the substrate in nm
        n0: refractive index of air
        wl: the incidence light wavelength in nm

    Returns:
        Tuple of M_pp and M_ss matrices
    """
    chi_sub_pp, chi_sub_ss = calc_chi_substrate(phi_deg, n_sub, n0)
    p_f_pp, p_f_ss = calc_p_matrices(phi_deg, n_f, d_f, n0, wl)
    chi_0_pp, chi_0_ss = calc_chi_ambient(phi_deg, n0)

    n_0f = bruggeman(n0, n_f)
    p_0f_pp, p_0f_ss = calc_p_matrices(phi_deg, n_0f, d_0f, n0, wl)

    n_fs = bruggeman(n_f, n_sub)
    p_fs_pp, p_fs_ss = calc_p_matrices(phi_deg, n_fs, d_fs, n0, wl)

    m_pp = chi_0_pp @ p_0f_pp @ p_f_pp @ p_fs_pp @ chi_sub_pp
    m_ss = chi_0_ss @ p_0f_ss @ p_f_ss @ p_fs_ss @ chi_sub_ss
    return (m_pp, m_ss)


@jax.jit
def calc_psi_delta_from_m_matrices(m_pp: Complex[Array, "2 2"],
                                   m_ss: Complex[Array, "2 2"]) -> tuple[Float, Float]:
    """
    Calculate the ellipsometric angles Psi and Delta from the characteristic matrices M_pp and M_ss
    (cf. (3.21a,b) in H. Tompkins, E. Irene, Handbook of ellipsometry, 2005)

    Args:
        m_pp: the characteristic matrix M_pp
        m_ss: the characteristic matrix M_ss

    Returns:
        Tuple of Psi and Delta in degrees
    """
    r_pp = m_pp[1][0] / m_pp[0][0]
    r_ss = m_ss[1][0] / m_ss[0][0]
    rho = r_pp / r_ss
    psi = jnp.arctan(abs(rho))
    psi_deg = jnp.degrees(psi)
    delta = jnp.angle(rho, deg=True)
    return (psi_deg, jnp.where(delta < 0, delta + 360, delta))


@jax.jit
def calc_psi_delta_one_layer(phi_deg: Float,
                             n_f: Complex,
                             d_f: Float,
                             n_sub: Complex,
                             n0: Complex = 1.0 + 0j,
                             wl: Float = 632.8) -> tuple[Float, Float]:
    """
    Compute the ellipsometric angles Psi and Delta for the layer stack consisting of
    one uniform film on the substrate

    Args:
        phi_deg: incidence angle of the light beam in ambient in degrees
        n_f: complex refractive index of the film
        d_f: the film thickness in nm
        n_sub: complex refractive index of the substrate
        n0: refractive index of air
        wl: the incidence light wavelength in nm

    Returns:
        Tuple of the Psi and Delta ellipsometric angles in degrees
    """
    m_pp, m_ss = calc_m_matrices_one_layer(phi_deg, n_f, d_f, n_sub, n0, wl)
    psi, delta = calc_psi_delta_from_m_matrices(m_pp, m_ss)
    return (psi, delta)


@jax.jit
def calc_psi_delta_one_layer_vec(phi_deg: Float[Array, 'len'],
                                 n_f: Complex,
                                 d_f: Float,
                                 n_sub: Complex,
                                 n0: Complex = 1.0 + 0j,
                                 wl: Float = 632.8) -> tuple[Float[Array, 'len'],
                                                             Float[Array, 'len']]:
    """
    Vectorized version of calc_psi_delta_one_layer function for array of incidence angles of
    the light beam in ambient.
    """
    return jax.vmap(calc_psi_delta_one_layer,
                    in_axes=[0,
                             None,
                             None,
                             None,
                             None,
                             None],
                    out_axes=0)(phi_deg, n_f, d_f, n_sub, n0, wl)


@jax.jit
def calc_psi_delta_three_layers(phi_deg: Float,
                                n_f: Complex,
                                d_f: Float,
                                n_sub: Complex,
                                d_0f: Float,
                                d_fs: Float,
                                n0: Complex = 1.0 + 0j,
                                wl: Float = 632.8) -> tuple[Float, Float]:
    """
    Compute the ellipsometric angles Psi and Delta for the layer stack consisting of
    three layers on the substrate, i.e. the uniform film layer, the interface layer between
    the film and the ambient, and the  interface layer between the film and the substrate.
    The refractive index of the interface layer is modelled
    with the Bruggeman EMA, assuming equal content of both adjacent phases.

    Args:
        phi_deg: incidence angle of the light beam in air in degrees
        n_f: complex refractive index of the film
        d_f: the film thickness in nm
        n_sub: complex refractive index of the substrate
        d_0f: interface width between the film and the ambient in nm
        d_fs: interface width between the film and the substrate in nm
        n0: refractive index of air
        wl: the incidence light wavelength in nm

    Returns:
        Tuple of the Psi and Delta ellipsometric angles in degrees
    """
    m_pp, m_ss = calc_m_matrices_three_layers(phi_deg, n_f, d_f, n_sub, d_0f, d_fs, n0, wl)
    psi, delta = calc_psi_delta_from_m_matrices(m_pp, m_ss)
    return (psi, delta)


@jax.jit
def calc_psi_delta_three_layers_vec(phi_deg: Float[Array, 'len'],
                                    n_f: Complex,
                                    d_f: Float,
                                    n_sub: Complex,
                                    d_0f: Float,
                                    d_fs: Float,
                                    n0: Complex = 1.0 + 0j,
                                    wl: Float = 632.8) -> tuple[Float[Array, 'len'],
                                                                Float[Array, 'len']]:
    """
    Vectorized version of calc_psi_delta_three_layers function for array of incidence angles of
    the light beam in ambient.
    """
    return jax.vmap(calc_psi_delta_three_layers,
                    in_axes=[0,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             None],
                    out_axes=0)(phi_deg, n_f, d_f, n_sub, d_0f, d_fs, n0, wl)
