import sympy as sy

"""
Summary of Symbolic Expressions for Astrophysical Calculations

This module contains symbolic expressions used in astrophysical calculations, particularly those described in Luminet (1977). Here's what each expression represents and why we use it:

1. expr_Q():
    - Represents the discriminant of a quartic equation.
    - Helps in determining the roots of an objective function, crucial for calculating impact parameters.

2. expr_radial_coordinate():
    - Symbolizes the radial coordinate (b) in the observer's frame.
    - Used to calculate the distance of isoradial images from the black hole, considering observer inclination.

3. expr_inverse_radial():
    - Symbolizes the inverse of the radial coordinate (1/r).
    - Helps in calculating periastron distances, essential for understanding image formation.

4. expr_argument_sn():
    - Represents the argument to a mathematical function called Jacobi elliptic function sn.
    - Needed for solving elliptic integrals, which describe the shape of isoradial images.

5. expr_gamma():
    - Symbolizes an angle (gamma) relating observer's and source's positions.
    - Important for understanding how the observer's inclination affects the observed images.

6. expr_k_squared():
    - Represents a parameter (k^2) used in elliptic integrals.
    - Required for calculating elliptic functions, which describe the shape of isoradial images.

7. expr_zeta_inf():
    - Symbolizes a parameter (zeta_inf) used in elliptic integrals.
    - Helps in determining the argument of elliptic functions, crucial for image calculations.

8. expr_ellipse():
    - Represents the mathematical expression for an ellipse.
    - Describes the shape of isoradial images viewed at different observer inclinations.

9. expr_objective():
    - Symbolizes an objective function used in root finding.
    - Helps in finding periastron distances corresponding to isoradials, aiding in image calculations.

10. expr_flux_accreting_disk():
    - Represents the radiation flux of an accreting disk.
    - Essential for understanding the emitted radiation from accreting material around a black hole.

11. expr_normalized_radial_coordinate():
    - Symbolizes the radial coordinate normalized by the black hole mass.
    - Facilitates comparisons across different black hole masses, aiding in data interpretation.

12. expr_redshift():
    - Represents the redshift effect observed in astronomical spectra.
    - Accounts for relativistic effects near a massive object like a black hole.

13. expr_bolometric_flux():
    - Represents the observed bolometric flux across all wavelengths.
    - Essential for understanding the total energy emitted by an accreting disk.

14. expr_normalized_bolometric_flux():
    - Symbolizes the normalized observed bolometric flux.
    - Helps in comparing observed fluxes across different astrophysical scenarios, considering black hole mass and accretion rate.

Words to Define:
- Isoradial: A term describing lines of constant radius from the central object, such as a black hole. In this context, it refers to the concentric circles or lines in the accretion disk around the black hole.
- Periastron: The point in the orbit of one body around another (e.g., a star around a black hole) that is closest to the center of the other body.
- Elliptic Integrals: Mathematical functions used to describe the arc length of an elliptic curve.
- Discriminant: A term in algebraic equations that helps determine the nature of the roots (e.g., real, imaginary) of the equation.

Each expression is documented with intuitive explanations to help users understand its purpose and significance in astrophysical calculations.
"""

def generate_Q_expression() -> sy.Symbol:
    """Generate a symbolic expression for Q.

    Returns
    -------
    sy.Symbol
        Symbolic expression representing Q
    """
    # Define symbols for periastron distance (periastron) and black hole mass (mass)
    periastron_distance, black_hole_mass = sy.symbols("periastron_distance black_hole_mass")

    # Calculate the expression for Q using the given formula
    Q_expression = sy.sqrt((periastron_distance - 2 * black_hole_mass) * (periastron_distance + 6 * black_hole_mass))

    return Q_expression

def generate_b_expression() -> sy.Symbol:
    """Generate a symbolic expression for b, the radial coordinate in the observer's frame.

    The expression is derived from equation 5 in Luminet (1977) with an adjustment for a dimensional error.

    Returns
    -------
    sy.Symbol
        Symbolic expression for b
    """
    # Define symbols for periastron distance (periastron) and black hole mass (mass)
    periastron_distance, black_hole_mass = sy.symbols("periastron_distance black_hole_mass")

    # Calculate the expression for b using the corrected formula
    # Dimensional analysis suggests taking the square root of P**3/(P-2M)
    b_expression = sy.sqrt((periastron_distance**3) / (periastron_distance - 2 * black_hole_mass))

    return b_expression

def generate_inverse_r_expression() -> sy.Symbol:
    """Generate a symbolic expression for 1/r.

    The expression is derived from equation 13 in Luminet (1977) with an adjustment for an algebra error.

    Returns
    -------
    sy.Symbol
        Symbolic expression for 1/r
    """
    # Define symbols for periastron distance (periastron), black hole mass (mass),
    # dimensionless periastron distance (q), dimensionless radial coordinate (u),
    # and the modulus of the elliptic function (k)
    periastron_distance, black_hole_mass, q, u, k = sy.symbols("periastron_distance black_hole_mass q u k")
    # Define elliptic function sn
    elliptic_function_sn = sy.Function("sn")

    # Calculate the expression for 1/r using the corrected formula
    # Equation 13 from Luminet (1977) has an algebra error, which is adjusted here
    # The corrected expression involves the elliptic function sn
    inverse_r_expression = (1 / (4 * black_hole_mass * periastron_distance)) * \
        (-(q - periastron_distance + 2 * black_hole_mass) + \
         (q - periastron_distance + 6 * black_hole_mass) * elliptic_function_sn(u, k**2) ** 2)

    return inverse_r_expression

def generate_argument_to_sn_expression() -> sy.Symbol:
    """Generate a symbolic expression for the argument to the elliptic function sn.

    The expression is derived from equation 13 in Luminet (1977) with an adjustment for an algebra error.

    Parameters
    ----------
    None

    Returns
    -------
    sy.Symbol
        Symbolic expression for the argument of sn
    """
    # Define symbols for gamma, zeta_inf, k, p, q, and n as per Luminet (1977)
    gamma, zeta_inf, k, p, q, n = sy.symbols("gamma zeta_inf k p q n")

    # Define the Piecewise expression to handle different cases in equation 13
    argument_to_sn_expression = sy.Piecewise(
        # Case when n = 0
        (
            gamma / (2 * sy.sqrt(p / q)) + sy.elliptic_f(zeta_inf, k**2),
            sy.Eq(n, 0)
        ),
        # Default case when n != 0
        (
            (gamma - 2 * n * sy.pi) / (2 * sy.sqrt(p / q)) -
            sy.elliptic_f(zeta_inf, k**2) + 2 * sy.elliptic_k(k**2),
            True
        )
    )

    return argument_to_sn_expression

def generate_gamma_expression() -> sy.Symbol:
    """Generate a symbolic expression for gamma, an angle that relates alpha and theta_0.

    The expression is derived from equation 10 of Luminet (1977).

    Returns
    -------
    sy.Symbol
        Symbolic expression for gamma
    """
    # Define symbols for alpha and theta_0 as per Luminet (1977)
    alpha, theta_0 = sy.symbols("alpha theta_0")

    # Define the expression for gamma using equation 10 from Luminet (1977)
    gamma_expression = sy.acos(
        sy.cos(alpha) / sy.sqrt(sy.cos(alpha) ** 2 + sy.tan(theta_0) ** -2)
    )

    return gamma_expression

def generate_k_expression() -> sy.Symbol:
    """Generate a symbolic expression for k, where k**2 is used as a modulus in the elliptic integrals.

    The expression is derived from equation 12 of Luminet (1977).

    Returns
    -------
    sy.Symbol
        Symbolic expression for k
    """
    # Define symbols for P, M, and Q as per Luminet (1977)
    p, m, q = sy.symbols("P M Q")

    # Define the expression for k using equation 12 from Luminet (1977)
    k_expression = sy.sqrt((q - p + 6 * m) / (2 * q))

    return k_expression

def generate_zeta_inf_expression() -> sy.Symbol:
    """Generate a symbolic expression for zeta_inf.

    The expression is derived from equation 12 of Luminet (1977).

    Returns
    -------
    sy.Symbol
        Symbolic expression for zeta_inf
    """
    # Define symbols for P, M, and Q as per Luminet (1977)
    p, m, q = sy.symbols("P M Q")

    # Define the expression for zeta_inf using equation 12 from Luminet (1977)
    zeta_inf_expression = sy.asin(sy.sqrt((q - p + 2 * m) / (q - p + 6 * m)))

    return zeta_inf_expression

def generate_ellipse_expression() -> sy.Symbol:
    """Generate a symbolic expression for an ellipse.

    In the Newtonian limit, isoradials form these images.

    Returns
    -------
    sy.Symbol
        Symbolic expression for an ellipse viewed at an inclination of theta_0
    """
    # Define symbols for r, alpha, and theta_0
    r, alpha, theta_0 = sy.symbols("r alpha theta_0")

    # Define the expression for the ellipse using the formula
    ellipse_expression = r / sy.sqrt(1 + (sy.tan(theta_0) ** 2) * (sy.cos(alpha) ** 2))

    return ellipse_expression

def generate_objective_function() -> sy.Symbol:
    """Generate a symbolic expression for the objective function.

    The objective function has roots which are periastron distances for isoradials.

    Returns
    -------
    sy.Symbol
        Symbolic expression for the objective function
    """
    # Define symbol for r
    r = sy.symbols("r")

    # Generate the expression for 1/r
    expression_r_inv = generate_inverse_r_expression()

    # Define the expression for the objective function using the formula
    objective_function = 1 - r * expression_r_inv

    return objective_function

def generate_flux_expression() -> sy.Symbol:
    """Generate an expression for the flux of an accreting disk.

    See equation 15 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Sympy expression for Fs, the radiation flux of an accreting disk
    """
    # Define symbols for the variables in the equation
    m, r_star, mdot = sy.symbols(r"M, r^*, \dot{m}")

    # Define the expression for the flux of the accreting disk using the given formula
    flux_expression = (
        ((3 * m * mdot) / (8 * sy.pi))
        * (1 / ((r_star - 3) * r_star ** (5 / 2)))
        * (
            sy.sqrt(r_star)
            - sy.sqrt(6)
            + (sy.sqrt(3) / 3)
            * sy.ln(
                ((sy.sqrt(r_star) + sy.sqrt(3)) * (sy.sqrt(6) - sy.sqrt(3)))
                / ((sy.sqrt(r_star) - sy.sqrt(3)) * (sy.sqrt(6) + sy.sqrt(3)))
            )
        )
    )

    return flux_expression

def generate_normalized_radial_expression() -> sy.Symbol:
    """Generate an expression for r^*, the radial coordinate normalized by the black hole mass.

    Returns
    -------
    sy.Symbol
        Sympy expression for the radial coordinate normalized by the black hole mass.
    """
    # Define symbols for the black hole mass (M) and radial coordinate (r)
    m, r = sy.symbols("M, r")

    # Define the expression for r* (normalized radial coordinate) as r/M
    normalized_radial_expression = r / m

    return normalized_radial_expression

def generate_redshift_expression() -> sy.Symbol:
    """Generate an expression for the redshift 1+z.

    See equation 19 in Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol()
        Sympy expression for the redshift of the accretion disk
    """
    # Define symbols for the black hole mass (M), radial coordinate (r), observer inclination (theta_0),
    # polar angle (alpha), and impact parameter (b)
    m, r, theta_0, alpha, b = sy.symbols("M, r, theta_0, alpha, b")

    # Define the expression for 1+z using the given formula from Luminet (1977)
    redshift_expression = (1 + sy.sqrt(m / r**3) * b * sy.sin(theta_0) * sy.sin(alpha)) / sy.sqrt(
        1 - 3 * m / r
    )

    return redshift_expression

def generate_bolometric_flux_expression() -> sy.Symbol:
    """Generate an expression for the observed bolometric flux.

    Returns
    -------
    sy.Symbol
        Sympy expression for the raw bolometric flux.
    """
    # Define symbols for the bolometric flux (Fs) and observed redshift (z_op)
    fs, opz = sy.symbols("F_s, z_op")

    # Define the expression for the observed bolometric flux using the given formula
    bolometric_flux_expression = fs / opz**4

    return bolometric_flux_expression

def generate_normalized_bolometric_flux_expression() -> sy.Symbol:
    """Generate an expression for the normalized observed bolometric flux.

    Units are in (8*pi)/(3*M*Mdot).

    Returns
    -------
    sy.Symbol
        Sympy expression for the normalized bolometric flux.
    """
    # Define symbols for the black hole mass (M) and accretion rate (\dot{m})
    m, mdot = sy.symbols(r"M, \dot{m}")

    # Calculate the raw bolometric flux expression
    raw_flux_expression = generate_bolometric_flux_expression()

    # Define the expression for the normalized observed bolometric flux
    normalized_flux_expression = raw_flux_expression / ((8 * sy.pi) / (3 * m * mdot))

    return normalized_flux_expression
