"""
Convergence_LineSearch
======================

Convergence checking and line search implementation for optimization algorithms.
Provides convergence criterion checking (KKT conditions) and Strong-Wolfe line search
for finding optimal step sizes.

Author: ASO Project Template
Date: 2026
"""

import logging
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def convergence_check(
    gradient, lagrange_multipliers, constraints=None,
    problem_m=0, problem_me=0,
    gradient_tol=1e-5, constraint_tol=1e-5, complementarity_tol=1e-5
):
    """
    Check convergence according to the first-order necessary (KKT) conditions.

    Verifies:
    1) Stationarity: max |grad| <= gradient_tol
    2) Primal feasibility: inequalities g_i <= constraint_tol, equalities |h_j| <= constraint_tol
    3) Dual feasibility: lambda_i >= -complementarity_tol (allow tiny negative noise)
    4) Complementarity: |lambda_i * g_i| <= complementarity_tol

    Parameters
    ----------
    gradient : numpy.ndarray
        Current gradient of the Lagrange function with respect to design variables.
    lagrange_multipliers : numpy.ndarray
        Current Lagrange multipliers [lambda_1, ..., lambda_m, mu_1, ..., mu_me].
    constraints : numpy.ndarray, optional
        Current constraint values [g_1, ..., g_m, h_1, ..., h_me].
    problem_m : int
        Number of inequality constraints.
    problem_me : int
        Number of equality constraints.
    gradient_tol : float, default: 1e-5
        Tolerance for gradient stationarity.
    constraint_tol : float, default: 1e-5
        Tolerance for constraint feasibility.
    complementarity_tol : float, default: 1e-5
        Tolerance for complementarity condition.

    Returns
    -------
    bool
        True if all KKT conditions are satisfied, False otherwise.
    """
    grad = np.asarray(gradient, dtype=float)

    # 1) Stationarity
    max_abs_grad = float(np.max(np.abs(grad))) if grad.size > 0 else 0.0
    stationarity_ok = max_abs_grad <= gradient_tol
    logger.debug("convergence_check: max|grad|={:.3e} tol={:.3e}".format(max_abs_grad, gradient_tol))

    # If no constraints provided, return stationarity result only
    if constraints is None:
        return bool(stationarity_ok)

    # Convert constraints and check length
    cons = np.asarray(constraints, dtype=float)
    m = int(problem_m)
    me = int(problem_me)
    expected = m + me
    if cons.size != expected:
        raise ValueError("constraints must have length m+me = {}, got {}".format(expected, cons.size))

    # Split constraints: first m are inequalities g(x) (should be <= 0),
    # last me are equalities h(x) (should be == 0)
    g_ineq = cons[:m] if m > 0 else np.empty(0)
    h_eq = cons[m:] if me > 0 else np.empty(0)

    # Lagrange multipliers
    lm = np.asarray(lagrange_multipliers, dtype=float)
    lm_ineq = lm[:m] if m > 0 else np.empty(0)
    lm_eq = lm[m:] if me > 0 else np.empty(0)

    # 2) Primal feasibility
    feasible_ineq = True if g_ineq.size == 0 else np.all(g_ineq <= constraint_tol)
    feasible_eq = True if h_eq.size == 0 else np.all(np.abs(h_eq) <= constraint_tol)

    # 3) Dual feasibility (inequality multipliers non-negative within tol)
    dual_feasible = True if lm_ineq.size == 0 else np.all(lm_ineq >= -complementarity_tol)

    # 4) Complementarity: lambda_i * g_i approximately zero
    complementarity = True if g_ineq.size == 0 else np.all(np.abs(lm_ineq * g_ineq) <= complementarity_tol)

    logger.debug(
        "convergence_check: feasible_ineq=%s feasible_eq=%s dual_feasible=%s complementarity=%s",
        feasible_ineq, feasible_eq, dual_feasible, complementarity,
    )

    return bool(stationarity_ok and feasible_ineq and feasible_eq and dual_feasible and complementarity)


def line_search(
    objective_func, grad_objective_func,
    x_current, direction,
    alpha_ini=1.0, alpha_min=1e-16, alpha_max=1.0,
    algorithm="STRONG_WOLFE",
    m1=1e-4, m2=0.1
):
    """
    Strong-Wolfe line search for computing optimal step size.

    Finds a step size α such that:
        1) Armijo (sufficient decrease):
           f(x + α*p) ≤ f(x) + c₁*α*∇f(x)ᵀ*p
        
        2) Strong Wolfe (curvature):
           |∇f(x + α*p)ᵀ*p| ≤ -c₂*∇f(x)ᵀ*p

    where:
        - c₁ = m1 (typically 1e-4)
        - c₂ = m2 (typically 0.1 for gradient methods)
        - p = search direction
        - ∇f = gradient of objective function

    Parameters
    ----------
    objective_func : callable
        Objective function f(x) returning float.
    grad_objective_func : callable
        Gradient function ∇f(x) returning NDArray.
    x_current : numpy.ndarray
        Current design variables.
    direction : numpy.ndarray
        Search direction (typically -∇f for steepest descent).
    alpha_ini : float, default: 1.0
        Initial step size to try.
    alpha_min : float, default: 1e-16
        Minimum allowable step size (machine precision).
    alpha_max : float, default: 1.0
        Maximum allowable step size.
    algorithm : str, default: "STRONG_WOLFE"
        Line search algorithm. Only "STRONG_WOLFE" is fully implemented.
    m1 : float, default: 1e-4
        Armijo constant c₁ (sufficient decrease tolerance).
    m2 : float, default: 0.1
        Curvature constant c₂ (must satisfy 0 < m1 < m2 < 1).

    Returns
    -------
    float
        Computed step size α ∈ [alpha_min, alpha_max] satisfying
        Armijo and Strong-Wolfe conditions.
    """
    
    # =========================================================================
    # PARAMETER VALIDATION
    # =========================================================================
    
    if not (0 < m1 < m2 < 1):
        raise ValueError(
            "Line search parameters must satisfy: 0 < m1 < m2 < 1, "
            "but got m1={}, m2={}".format(m1, m2)
        )
    if alpha_min >= alpha_max:
        raise ValueError("alpha_min ({}) must be < alpha_max ({})".format(alpha_min, alpha_max))
    
    if algorithm not in ("STRONG_WOLFE", "WOLFE"):
        logger.warning(
            "Line search algorithm '%s' not fully implemented. Using STRONG_WOLFE.", 
            algorithm
        )
    
    # =========================================================================
    # INITIALIZE LINE SEARCH
    # =========================================================================
    
    x0 = x_current.copy()
    
    # φ(α) = f(x₀ + α*p) - univariate function of step size
    def phi(alpha: float) -> float:
        """Objective function value at x₀ + α*p."""
        alpha_clipped = max(alpha_min, min(alpha, alpha_max))
        x_trial = x0 + alpha_clipped * direction
        return float(objective_func(x_trial))
    
    # φ'(α) = ∇f(x₀ + α*p)ᵀ * p - directional derivative
    def phi_prime(alpha: float) -> float:
        """Directional derivative at x₀ + α*p along direction p."""
        alpha_clipped = max(alpha_min, min(alpha, alpha_max))
        x_trial = x0 + alpha_clipped * direction
        grad = grad_objective_func(x_trial)
        return float(np.dot(grad, direction))
    
    # Initial values at α = 0
    phi_0 = phi(0.0)
    phi_prime_0 = phi_prime(0.0)
    
    # =========================================================================
    # ENSURE DESCENT DIRECTION
    # =========================================================================
    
    if phi_prime_0 >= 0:
        logger.warning(
            "Line search: direction not descent (dir_deriv=%.2e >= 0). "
            "Reversing direction.", 
            phi_prime_0
        )
        direction = -direction
        phi_prime_0 = -phi_prime_0
        
        if phi_prime_0 >= 0:
            logger.warning("Line search: even reversed direction is not descent. Returning minimal step.")
            return alpha_min
    
    # =========================================================================
    # BRACKETING PHASE
    # =========================================================================
    
    alpha_prev = 0.0
    phi_prev = phi_0
    phi_prime_prev = phi_prime_0
    
    alpha = max(alpha_min, min(alpha_ini, alpha_max))
    
    max_bracket_iter = 50
    
    for bracket_iter in range(max_bracket_iter):
        
        phi_alpha = phi(alpha)
        phi_prime_alpha = phi_prime(alpha)
        
        # ARMIJO CONDITION
        armijo_violated = phi_alpha > phi_0 + m1 * alpha * phi_prime_0
        
        if armijo_violated or (bracket_iter > 0 and phi_alpha >= phi_prev):
            return _line_search_zoom(
                alpha_prev, alpha, phi_0, phi_prime_0, 
                m1, m2, phi, phi_prime
            )
        
        # STRONG-WOLFE CURVATURE
        if abs(phi_prime_alpha) <= -m2 * phi_prime_0:
            return max(alpha_min, min(alpha, alpha_max))
        
        if phi_prime_alpha >= 0:
            return _line_search_zoom(
                alpha, alpha_prev, phi_0, phi_prime_0,
                m1, m2, phi, phi_prime
            )
        
        alpha_prev = alpha
        phi_prev = phi_alpha
        phi_prime_prev = phi_prime_alpha
        
        alpha = min(2.0 * alpha, alpha_max)
        
        if alpha >= alpha_max:
            return _line_search_zoom(
                alpha_prev, alpha, phi_0, phi_prime_0,
                m1, m2, phi, phi_prime
            )
    
    logger.warning("Line search bracket: max iterations reached, returning best estimate")
    return max(alpha_min, min(alpha_prev, alpha_max))


def _line_search_zoom(
    alpha_lo, alpha_hi, phi_0, phi_prime_0,
    m1, m2, phi, phi_prime
):
    """
    Zoom phase of Strong-Wolfe line search.

    Refines the bracketing interval [alpha_lo, alpha_hi] to find a step
    size satisfying both Armijo and Strong-Wolfe conditions.

    Parameters
    ----------
    alpha_lo, alpha_hi : float
        Bracketing interval bounds
    phi_0 : float
        Objective value at α=0
    phi_prime_0 : float
        Directional derivative at α=0
    m1, m2 : float
        Wolfe constants
    phi : callable
        Function φ(α) = f(x₀ + α*p)
    phi_prime : callable
        Function φ'(α) = ∇f(x₀ + α*p)ᵀ*p

    Returns
    -------
    float
        Step size satisfying convergence criteria
    """
    
    max_zoom_iter = 40
    
    for zoom_iter in range(max_zoom_iter):
        
        alpha = _line_search_interpolate(
            alpha_lo, phi(alpha_lo), phi_prime(alpha_lo),
            alpha_hi, phi(alpha_hi), phi_prime(alpha_hi)
        )
        
        margin = 1e-14 * (abs(alpha_hi) + 1.0)
        if alpha <= alpha_lo + margin or alpha >= alpha_hi - margin:
            alpha = 0.5 * (alpha_lo + alpha_hi)
        
        phi_alpha = phi(alpha)
        
        # CHECK ARMIJO
        if (phi_alpha > phi_0 + m1 * alpha * phi_prime_0) or (phi_alpha >= phi(alpha_lo)):
            alpha_hi = alpha
        else:
            phi_prime_alpha = phi_prime(alpha)
            
            # CHECK STRONG-WOLFE
            if abs(phi_prime_alpha) <= -m2 * phi_prime_0:
                return alpha
            
            if phi_prime_alpha * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            
            alpha_lo = alpha
        
        if abs(alpha_hi - alpha_lo) < 1e-14:
            break
    
    return alpha_lo


def _line_search_interpolate(a, fa, fpa, b, fb, fpb):
    """
    Cubic interpolation for line search refinement.

    Computes the minimizer of a cubic polynomial p(t) such that:
        p(a) = fa,   p'(a) = fpa
        p(b) = fb,   p'(b) = fpb

    Falls back to bisection if cubic interpolation fails.

    Parameters
    ----------
    a, fa, fpa : float
        Lower bound, function value, and derivative at lower bound
    b, fb, fpb : float
        Upper bound, function value, and derivative at upper bound

    Returns
    -------
    float
        Estimated minimizer, typically in the interior of [a, b]
    """
    
    d = b - a
    
    if abs(d) < 1e-16:
        return 0.5 * (a + b)
    
    try:
        C = fa
        D = fpa
        E = (3.0 * (fb - fa) / d - 2.0 * fpa - fpb) / d
        F = (2.0 * (fa - fb) / d + fpa + fpb) / (d * d)
        
        if abs(F) < 1e-20:
            if abs(E) < 1e-20:
                return 0.5 * (a + b)
            s = -D / (2.0 * E)
        else:
            disc = 4.0 * E * E - 12.0 * F * D
            if disc < 0:
                return 0.5 * (a + b)
            
            sqrt_disc = np.sqrt(disc)
            s1 = (-2.0 * E + sqrt_disc) / (6.0 * F)
            s2 = (-2.0 * E - sqrt_disc) / (6.0 * F)
            
            s = None
            for s_candidate in (s1, s2):
                if 0 < s_candidate < 1:
                    s = s_candidate
                    break
            
            if s is None:
                return 0.5 * (a + b)
        
        t = a + s * d
        return max(min(t, b), a)
    
    except (ValueError, OverflowError):
        return 0.5 * (a + b)
