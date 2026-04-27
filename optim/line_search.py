from scipy.optimize import minimize_scalar


def brent_line_search(f, bracket: float = 4.0, tol: float = 1e-5, max_iter: int = 50) -> float:
    """
    Find t* = argmin_{t in [0, bracket]} f(t) using Brent's method.

    Brent fits a parabola through function evaluations and jumps to its
    minimum, falling back to golden section when the quadratic fit is
    unreliable.  This makes it the natural analogue of the analytic exact
    line search (t* = ||g||^2 / g^T H g) used for quadratics: it recovers
    that answer exactly for quadratic objectives and degrades gracefully for
    nonlinear ones.  Typically needs ~10-15 evaluations vs ~30 for pure
    golden section.

    Args:
        f:       scalar function of t (forward pass only, no gradients needed)
        bracket: upper bound of the search interval [0, bracket]
        tol:     absolute tolerance on t*
        max_iter: maximum function evaluations
    """
    result = minimize_scalar(
        f,
        bounds=(0.0, bracket),
        method="bounded",
        options={"xatol": tol, "maxiter": max_iter},
    )
    return float(result.x)
