import multiprocessing as mp
from typing import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.special as sp
import sympy as sy

from luminet.functions import simulate_flux
from luminet.expressions import generate_redshift_expression
from luminet.utils import *

def generate_blackhole_data(
    alpha: npt.NDArray[np.float64],
    r_vals: Iterable[float],
    theta_0: float,
    n_vals: Iterable[int],
    m: float,
    root_kwargs,
) -> pd.DataFrame:
    """Generate the data needed to produce an image of a black hole.

    Parameters
    ----------
    alpha : npt.NDArray[np.float64]
        Alpha values at which the bolometric flux is to be simulated; angular coordinate in the
        observer's frame of reference
    r_vals : Iterable[float]
        Orbital radius of a section of the accretion disk from the center of the black hole
    theta_0 : float
        Inclination of the observer, in radians, with respect to the the normal of the accretion
        disk
    n_vals : Iterable[int]
        Order of the calculation; n=0 corresponds to the direct image, n>0 are ghost images
    m : float
        Mass of the black hole
    root_kwargs : Dict
        All other kwargs are passed to the `impact_parameter` function

    Returns
    -------
    pd.DataFrame
        Simulated data; columns are alpha, b, 1+z, r, n, flux, x, y
    """
    a_arrs = []
    b_arrs = []
    opz_arrs = []
    r_arrs = []
    n_arrs = []
    flux_arrs = []

    opz = sy.lambdify(["alpha", "b", "theta_0", "M", "r"], generate_redshift_expression())

    for n in n_vals:
        with mp.Pool(mp.cpu_count()) as pool:
            args = [(alpha, r, theta_0, n, m, None, root_kwargs) for r in r_vals]
            for r, (alpha_reoriented, b, opz, flux) in zip(
                r_vals, pool.starmap(simulate_flux, args)
            ):
                a_arrs.append(alpha_reoriented)
                b_arrs.append(b)
                opz_arrs.append(opz)
                r_arrs.append(np.full(b.size, r))
                n_arrs.append(np.full(b.size, n))
                flux_arrs.append(flux)

    df = pd.DataFrame(
        {
            "alpha": np.concatenate(a_arrs),
            "b": np.concatenate(b_arrs),
            "opz": np.concatenate(opz_arrs),
            "r": np.concatenate(r_arrs),
            "n": np.concatenate(n_arrs),
            "flux": np.concatenate(flux_arrs),
        }
    )

    df["x"] = df["b"] * np.cos(df["alpha"])
    df["y"] = df["b"] * np.sin(df["alpha"])
    return df
