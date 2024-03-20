from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import scipy.special as sp
import sympy as sy

import luminet.utils as util
from luminet.expressions import (generate_ellipse_expression,
                                    generate_argument_to_sn_expression,
                                    generate_b_expression,
                                    generate_bolometric_flux_expression,
                                    generate_flux_expression,
                                    generate_gamma_expression,
                                    #generate_inverse_r_expression,
                                    generate_k_expression,
                                    #generate_normalized_bolometric_flux_expression,
                                    generate_normalized_radial_expression,
                                    generate_objective_function,
                                    generate_Q_expression,
                                    generate_redshift_expression,
                                    generate_zeta_inf_expression)

def impact_parameter(
    alpha: np.typing.NDArray[float],
    r_value: float,
    theta_0: float,
    n: int,
    m: float,
    objective_func: Optional[Callable] = None,
    **root_kwargs
) -> np.typing.NDArray[float]:
    """Calculate the impact parameter for each value of alpha.

    Parameters
    ----------
    alpha : npt.NDArray[np.float64]
        Polar angle in the observer's frame of reference
    r_value : float
        Isoradial distance for which the impact parameter is to be calculated
    theta_0 : float
        Inclination of the observer with respect to the accretion disk plane normal
    n : int
        Order of the calculation; n=0 corresponds to the direct image, n>0 are ghost images
    m : float
        Mass of the black hole
    objective_func : Optional[Callable]
        Objective function whose roots are the periastron distances for a given b(alpha)
    root_kwargs
        Additional arguments are passed to fast_root

    Returns
    -------
    npt.NDArray[np.float64]
        Impact parameter b for each value of alpha. If no root of the objective function for the
        periastron is found for a particular value of alpha, the value of an ellipse is used at that
        point.
    """
    if objective_func is None:
        objective_func = lambda_objective()

    ellipse = lambdify(["r", "alpha", "theta_0"], generate_ellipse_expression())
    b = lambdify(["P", "M"], generate_b_expression())

    p_arr = util.fast_root(
        objective_func,
        np.linspace(2.1, 50, 1000),
        alpha,
        [theta_0, r_value, n, m],
        **root_kwargs
    )
    return np.where(np.isnan(p_arr), ellipse(r_value, alpha, theta_0), b(p_arr, m))

def reorient_alpha(alpha: Union[float, np.typing.NDArray[float]], n: int) -> float:
    """Reorient the polar angle on the observation coordinate system.

    From Luminet's paper:

        "...the observer will detect generally two images, a direct (or primary) image at polar
        coordinates (b^(d), alpha) and a ghost (or secundary) image at (b^(g), alpha + pi)."

    This function adds pi to the polar angle for ghost images, and returns the original angle for
    direct images.

    Parameters
    ----------
    alpha : float
        Polar angle alpha in the observer's "sensor" coordinate system.
    n : int
        Order of the image which is being calculated. n=0 corresponds to the direct image, while n>0
        corresponds to ghost images.

    Returns
    -------
    float
        Reoriented polar angle
    """
    return np.where(np.asarray(n) > 0, (alpha + np.pi) % (2 * np.pi), alpha)

def lambdify(*args, **kwargs) -> Callable:
    """Lambdify a sympy expression from Luminet's paper.

    Luminet makes use of the sn function, which is one of as Jacobi's elliptic functions. Sympy
    doesn't (yet) support this function, so lambdifying it requires specifying the correct scipy
    routine.

    Arguments are passed diretly to sympy.lambdify; if "modules" is specified, the user must specify
    which function to call for 'sn'.

    Parameters
    ----------
    *args
        Arguments are passed to sympy.lambdify
    **kwargs
        Additional kwargs passed to sympy.lambdify

    Returns
    -------
    Callable
        Lambdified expression
    """
    kwargs["modules"] = kwargs.get(
        "modules",
        [
            "numpy",
            {
                "sn": lambda u, m: sp.ellipj(u, m)[0],
                "elliptic_f": lambda phi, m: sp.ellipkinc(phi, m),
                "elliptic_k": lambda m: sp.ellipk(m),
            },
        ],
    )
    return sy.lambdify(*args, **kwargs)

def lambda_objective() -> Callable[[float, float, float, float, int, float], float]:
    """Generate a lambdified objective function.

    Returns
    -------
    Callable[(float, float, float, float, int, float), float]
        Objective function whose roots yield periastron distances for isoradials. The function
        signature is

            s(P, alpha, theta_0, r, n, m)
    """
    s = (
        generate_objective_function()
        .subs({"u": generate_argument_to_sn_expression()})
        .subs({"zeta_inf": generate_zeta_inf_expression()})
        .subs({"gamma": generate_gamma_expression()})
        .subs({"k": generate_k_expression()})
        .subs({"Q": generate_Q_expression()})
    )
    return lambdify(("P", "alpha", "theta_0", "r", "N", "M"), s)

def lambda_normalized_bolometric_flux() -> Callable[[float, float, float], float]:
    """Generate the normalized bolometric flux function.

    See `generate_image` for an example of how to use this.

    Returns
    -------
    Callable[(float, float, float), float]
        The returned function takes (1+z, r, M) as arguments and outputs the normalized bolometric
        flux of the black hole.
    """
    return sy.lambdify(
        ("z_op", "r", "M"),
        (
            generate_bolometric_flux_expression()
            .subs({"F_s": generate_flux_expression()})
            .subs({"M": 1, r"\dot{m}": 1})
            .subs({"r^*": generate_normalized_radial_expression()})
        )
        / (3 / (8 * sy.pi)),
    )

def simulate_flux(
    alpha: np.typing.NDArray[float],
    r: float,
    theta_0: float,
    n: int,
    m: float,
    objective_func: Optional[Callable] = None,
    root_kwargs: Optional[Dict[Any, Any]] = None,
) -> Tuple[
    np.typing.NDArray[float],
    np.typing.NDArray[float],
    np.typing.NDArray[float],
    np.typing.NDArray[float],
]:
    """Simulate the bolometric flux for an accretion disk near a black hole.

    Parameters
    ----------
    alpha : npt.NDArray[np.float64]
        Polar angle in the observer's frame of reference
    r : float
        Isoradial distance for which the impact parameter is to be calculated
    theta_0 : float
        Inclination of the observer with respect to the accretion disk plane normal
    n : int
        Order of the calculation; n=0 corresponds to the direct image, n>0 are ghost images
    m : float
        Mass of the black hole
    root_kwargs : Dict[Any, Any]
        Additional arguments are passed to fast_root
    objective_func : Optional[Callable]
        Objective function whose roots are the periastron distances for a given b(alpha)

    Returns
    -------
    Tuple[npt.NDArray[np.float64], ...]
        reoriented alpha, b, 1+z, and observed bolometric flux
    """
    flux = lambda_normalized_bolometric_flux()
    one_plus_z = sy.lambdify(["alpha", "b", "theta_0", "M", "r"], generate_redshift_expression())
    root_kwargs = root_kwargs if root_kwargs else {}

    b = impact_parameter(alpha, r, theta_0, n, m, objective_func, **root_kwargs)
    opz = one_plus_z(alpha, b, theta_0, m, r)

    return reorient_alpha(alpha, n), b, opz, flux(opz, r, m)
