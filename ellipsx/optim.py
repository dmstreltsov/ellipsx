"""
Ellipsometric model optimization
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex, Int

import optimistix as optx

from .models import calc_psi_delta_one_layer_vec
from .models import calc_psi_delta_one_layer_roughess_vec


@jax.jit
def residuals_one_layer_df_nf(params: Float[Array, "2"],
                              args: tuple[Float[Array, "5 len"],
                                          Float[Array, "2"],
                                          Float[Array, "2"],
                                          Float]) -> Float[Array, "len + len"]:
    """
    Compute residuals for the one layer model with thickness and real refractive index of
    the film as optimization parameters. The other model parameters are fixed.

    Args:
        params: array of variable parameters, i.e. film thickness, params[0],
                and real part of the film refractive index, n, params[1].
                The imag. part of the film refractive index, k, is assumed to be 0.
        args: tuple of the fixed parameters in the optimization, i.e.
              args[0]: matrix of phi, psi_obs, psi_std, delta_obs, delta_std vectors in degrees,
                       where phi is incidence angle of the light beam in ambient,
                             psi_obs is the observed value of Psi ellipsometric angle,
                             psi_std is the estimated standard deviation of psi_obs,
                             delta_obs is the observed value of Delta ellipsometric angle,
                             delta_std is the estimated standard deviation of delta_obs
              args[1]: real and imag. parts of the ambient refractive index
              args[2]: real and imag. parts of the substrate refractive index
              args[3]: wavelength of the light beam in ambient in nm

    Returns:
        Vector of weighted residuals for Psi and Delta for each incidence angle
    """

    d = params[0]
    n_f = params[1] + 0j  # imaginary part of the film refractive index is zero

    data, n0, n_sub, wl = args

    phi = data[0]
    psi_obs = data[1]
    psi_std = data[2]
    delta_obs = data[3]
    delta_std = data[4]

    n = jnp.array([n0[0] + 1j * n0[1],
                   n_f,
                   n_sub[0] + 1j * n_sub[1]])

    psi, delta = calc_psi_delta_one_layer_vec(phi, n, d, wl)

    return jnp.concatenate([(psi - psi_obs) / psi_std,
                            (delta - delta_obs) / delta_std])


@jax.jit
def one_layer_df_nf_lm(data: Float[Array, "5 len"],
                       params0: Float[Array, "2"],
                       n_sub: Complex,
                       n0: Complex = 1.0 + 0j,
                       wl: Float = 632.8,
                       rtol: Float = 1e-8,
                       atol: Float = 1e-8,
                       max_steps: Int = 1024) -> Float[Array, "2"]:
    """
    Optimize the single layer model using the nonlinear least squares optimization
    by the Levenberg-Marquardt method.
    The thickness in nm and refractive index of the layer are optimization parameters.
    The layer refractive index is assumed to be real, not complex.
    The other model parameters are fixed.

    Args:
        data: matrix of phi, psi_obs, psi_std, delta_obs, delta_std vectors in degrees,
              where phi is incidence angle of the light beam in ambient,
                    psi_obs is the observed value of Psi ellipsometric angle,
                    psi_std is the estimated standard deviation of psi_obs,
                    delta_obs is the observed value of Delta ellipsometric angle,
                    delta_std is the estimated standard deviation of delta_obs
        params0: initial guess for the layer thickness in nm, params[0], and
                 the layer refractive index, params[1]
        n_sub: the substrate refractive index, complex
        n0: the ambient refractive index, complex
        wl: the light beam wavelength in the ambient in nm
        rtol: relative tolerance for the Levenberg-Marquardt algorithm
        atol: absolute tolerance for the Levenberg-Marquardt algorithm
        max_steps: maximum number of steps for the least squares optimization

    Returns:
        sol_value: Array of the optimized values for film thickness, sol_value[0], and
                   film real refractive index, sol_value[1]
    """

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    residuals = residuals_one_layer_df_nf
    args = (data,
            jnp.array([n0.real, n0.imag]),
            jnp.array([n_sub.real, n_sub.imag]),
            wl)
    sol = optx.least_squares(residuals, solver, params0, args, max_steps=max_steps)

    return sol.value


@jax.jit
def one_layer_df_nf_params_std(data: Float[Array, "5 len"],
                               params_opt: Float[Array, "2"],
                               n_sub: Complex,
                               n0: Complex = 1.0 + 0j,
                               wl: Float = 632.8) -> tuple[Float[Array, "2"],
                                                           Float[Array, "2 2"]]:
    """
    Compute standard deviations and correlation matrix for the optimized parameters for
    one_layer_df_nf_lm function using linear approximation of the nonlinear least squares.

    Args:
        data: matrix of phi, psi_obs, psi_std, delta_obs, delta_std vectors in degrees,
              where phi is incidence angle of the light beam in ambient,
                    psi_obs is the observed value of Psi ellipsometric angle,
                    psi_std is the estimated standard deviation of psi_obs,
                    delta_obs is the observed value of Delta ellipsometric angle,
                    delta_std is the estimated standard deviation of delta_obs
        params_opt: optimized values for the layer thickness in nm, and
                    the layer refractive index, the output of one_layer_df_nf_lm function
        n_sub: the substrate refractive index, complex
        n0: the ambient refractive index, complex
        wl: the light beam wavelength in the ambient in nm

    Returns:
        (params_std, cor_matrix): standard deviations and correlation matrix for
                                  the optimized parameters
    """

    residuals = residuals_one_layer_df_nf
    args = (data,
            jnp.array([n0.real, n0.imag]),
            jnp.array([n_sub.real, n_sub.imag]),
            wl)

    jac_residuals = jax.jacobian(residuals)(params_opt, args)

    cov_matrix = jnp.linalg.inv(jac_residuals.T @ jac_residuals)  # parameter covariance matrix
    params_std = jnp.sqrt(jnp.diag(cov_matrix))  # parameter standard deviations
    cor_matrix = cov_matrix / jnp.outer(params_std, params_std)  # parameter correlation matrix

    return (params_std, cor_matrix)


@jax.jit
def residuals_one_layer_roughness_d_nf(params: Float[Array, "4"],
                                       args: tuple[Float[Array, "5 len"],
                                                   Float[Array, "2"],
                                                   Float[Array, "2"],
                                                   Float]) -> Float[Array, "len + len"]:
    """
    Compute residuals for the one layer model with interface roughness.
    Thicknesses of the layer and two interfaces, as well as real refractive index of
    the film as optimization parameters. The other model parameters are fixed.

    Args:
        params: array of variable parameters, i.e.
                ambient-film interface width in nm, params[0],
                film thickness in nm, params[1],
                film-substrate interface width in nm, params[2],
                real part of the film refractive index, n_f, params[3]. The imag. part of
                    the film refractive index, k, is assumed to be 0.
        args: tuple of the fixed parameters in the optimization, i.e.
              args[0]: matrix of phi, psi_obs, psi_std, delta_obs, delta_std vectors in degrees,
                       where phi is incidence angle of the light beam in ambient,
                             psi_obs is the observed value of Psi ellipsometric angle,
                             psi_std is the estimated standard deviation of psi_obs,
                             delta_obs is the observed value of Delta ellipsometric angle,
                             delta_std is the estimated standard deviation of delta_obs
              args[1]: real and imag. parts of the ambient refractive index
              args[2]: real and imag. parts of the substrate refractive index
              args[3]: wavelength of the light beam in ambient in nm

    Returns:
        Vector of weighted residuals for Psi and Delta for each incidence angle
    """

    d = jnp.array([params[0], params[1], params[2]])  # thicknesses of the three layers
    n_f = params[3] + 0j  # imaginary part of the film refractive index is zero

    data, n0, n_sub, wl = args

    phi = data[0]
    psi_obs = data[1]
    psi_std = data[2]
    delta_obs = data[3]
    delta_std = data[4]

    n = jnp.array([n0[0] + 1j * n0[1],
                   n_f,
                   n_sub[0] + 1j * n_sub[1]])

    psi, delta = calc_psi_delta_one_layer_roughess_vec(phi, n, d, wl)

    return jnp.concatenate([(psi - psi_obs) / psi_std,
                            (delta - delta_obs) / delta_std])


@jax.jit
def one_layer_roughness_d_nf_lm(data: Float[Array, "5 len"],
                                params0: Float[Array, "4"],
                                n_sub: Complex,
                                n0: Complex = 1.0 + 0j,
                                wl: Float = 632.8,
                                rtol: Float = 1e-8,
                                atol: Float = 1e-8,
                                max_steps: Int = 1024) -> Float[Array, "4"]:
    """
    Optimize the one layer model with interface roughness using the nonlinear least squares
    optimization by the Levenberg-Marquardt method.
    The thicknesses in nm of the layer and two interfaces, as well as refractive index of the layer
    are optimization parameters.
    The layer refractive index is assumed to be real, not complex.
    The other model parameters are fixed.

    Args:
        data: matrix of phi, psi_obs, psi_std, delta_obs, delta_std vectors in degrees,
              where phi is incidence angle of the light beam in ambient,
                    psi_obs is the observed value of Psi ellipsometric angle,
                    psi_std is the estimated standard deviation of psi_obs,
                    delta_obs is the observed value of Delta ellipsometric angle,
                    delta_std is the estimated standard deviation of delta_obs
        params0: initial guess for the ambient-film interface thickness in nm, params[0],
                 the film thickness in nm, params[1], the film-substrate interface thickness
                 in nm, params[2], as well as the film real refractive index, params[3]
        n_sub: the substrate refractive index, complex
        n0: the ambient refractive index, complex
        wl: the light beam wavelength in the ambient in nm
        rtol: relative tolerance for the Levenberg-Marquardt algorithm
        atol: absolute tolerance for the Levenberg-Marquardt algorithm
        max_steps: maximum number of steps for the least squares optimization

    Returns:
        sol_value: Array of the optimized values for ambient-film interface thickness,
                   sol_value[0], the film thickness, sol_value[1], the film-substrate
                   interface thickness, sol_value[2], as well as the film
                   real refractive index, sol_value[3]
    """

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    residuals = residuals_one_layer_roughness_d_nf
    args = (data,
            jnp.array([n0.real, n0.imag]),
            jnp.array([n_sub.real, n_sub.imag]),
            wl)
    sol = optx.least_squares(residuals, solver, params0, args, max_steps=max_steps)

    return sol.value


@jax.jit
def one_layer_roughness_d_nf_params_std(data: Float[Array, "5 len"],
                                        params_opt: Float[Array, "4"],
                                        n_sub: Complex,
                                        n0: Complex = 1.0 + 0j,
                                        wl: Float = 632.8) -> tuple[Float[Array, "4"],
                                                                    Float[Array, "4 4"]]:
    """
    Compute standard deviations and correlation matrix for the optimized parameters for
    one_layer_roughness_d_nf_lm function using linear approximation of the nonlinear least squares.

    Args:
        data: matrix of phi, psi_obs, psi_std, delta_obs, delta_std vectors in degrees,
              where phi is incidence angle of the light beam in ambient,
                    psi_obs is the observed value of Psi ellipsometric angle,
                    psi_std is the estimated standard deviation of psi_obs,
                    delta_obs is the observed value of Delta ellipsometric angle,
                    delta_std is the estimated standard deviation of delta_obs
        params_opt: optimized values for the ambient-film interface thickness in nm, params[0],
                    the film thickness in nm, params[1], the film-substrate interface thickness
                    in nm, params[2], as well as the film real refractive index, params[3],
                    the output of one_layer_roughness_d_nf_lm function
        n_sub: the substrate refractive index, complex
        n0: the ambient refractive index, complex
        wl: the light beam wavelength in the ambient in nm

    Returns:
        (params_std, cor_matrix): standard deviations and correlation matrix for
                                  the optimized parameters
    """

    residuals = residuals_one_layer_roughness_d_nf
    args = (data,
            jnp.array([n0.real, n0.imag]),
            jnp.array([n_sub.real, n_sub.imag]),
            wl)

    jac_residuals = jax.jacobian(residuals)(params_opt, args)

    cov_matrix = jnp.linalg.inv(jac_residuals.T @ jac_residuals)  # parameter covariance matrix
    params_std = jnp.sqrt(jnp.diag(cov_matrix))  # parameter standard deviations
    cor_matrix = cov_matrix / jnp.outer(params_std, params_std)  # parameter correlation matrix

    return (params_std, cor_matrix)
