"""
        Steepest-descent algorithm for unconstrained optimisation.

        Implements the classical steepest descent method:
            x(k+1) = x(k) + alpha(k) * p(k)
        where:
            p(k) = -grad(f(x(k)))  (search direction: negative gradient)
            alpha(k) = optimal step size found by line search

        The algorithm terminates when:
            ||grad(f(x(k)))|| <= gradient_tol  (convergence criterion)
        or when the iteration limit is reached.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of iterations allowed.
        gradient_tol : float, default: 1e-5
            Tolerance for gradient norm convergence criterion.
            Convergence occurs when max(|grad_f_i(x)|) <= gradient_tol.

        Returns
        -------
        int
            Number of iterations performed if converged, -1 if iteration limit
            exceeded without convergence.

        Author: ASO Project Template
        Date: 2026
"""

import logging

import numpy as np
from numpy.typing import NDArray

from Convergence_LineSearch import line_search
from Problems import get_problem

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def steepest_descent(
    problem_name, x_initial,
    iteration_limit=1000, gradient_tol=1e-5
):
    """
    Steepest-descent algorithm for unconstrained optimization.

    Parameters
    ----------
    problem_name : str
        Name of the problem to solve (e.g., "rosenbrock", "matyas").
    x_initial : NDArray
        Initial design variables.
    iteration_limit : int, default: 1000
        Maximum number of iterations allowed.
    gradient_tol : float, default: 1e-5
        Tolerance for gradient norm convergence criterion.
        Convergence occurs when max(|grad_f_i(x)|) <= gradient_tol.

    Returns
    -------
    tuple
        (iteration_count, x_final) where:
        - iteration_count: Number of iterations performed (-1 if limit exceeded)
        - x_final: Final design variables at convergence or iteration limit
    """
    
    # Load problem
    problem = get_problem(problem_name)
    
    # Copy initial point to avoid modifying input
    x = x_initial.copy()
    
    logger.info("Starting Steepest Descent for problem: {}".format(problem.name))
    logger.info("Initial point: {}".format(x))
    logger.info("Initial objective value: {:.6e}".format(problem.objective(x)))
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    # Validate parameters
    if gradient_tol <= 0:
        raise ValueError("gradient_tol must be positive.")
    
    iteration = 0

    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================
    
    while iteration < iteration_limit:
        
        # STEP 1: Compute gradient grad(f(x(k)))
        # ======================================
        grad_f = problem.grad_objective(x)
        grad_norm_inf = float(np.max(np.abs(grad_f)))
        grad_norm_2 = float(np.linalg.norm(grad_f))
        
        # STEP 2: Check convergence criterion: ||grad(f(x))|| <= tol
        # ==========================================================
        if grad_norm_inf <= gradient_tol:
            logger.info(
                "Steepest Descent converged after {} iterations. "
                "Max gradient component: {:.2e} <= tolerance {:.2e}".format(iteration, grad_norm_inf, gradient_tol)
            )
            f_final = problem.objective(x)
            logger.info("Final objective value: {:.6e}".format(f_final))
            return iteration, x
        
        # STEP 3: Compute search direction p(k) = -grad(f(x(k)))
        # ========================================================
        # The negative gradient is the direction of steepest descent.
        # It points in the direction of maximum decrease of f.
        p = -grad_f
        
        # STEP 4: Find optimal step size alpha(k) using line search
        # ==========================================================
        # Use Strong-Wolfe line search for robust convergence.
        # This ensures sufficient decrease (Armijo) and appropriate curvature.
        alpha = line_search(
            objective_func=problem.objective,
            grad_objective_func=problem.grad_objective,
            x_current=x,
            direction=p,
            alpha_ini=1.0,          # Initial step size candidate
            alpha_min=1e-16,        # Minimum allowable step size
            alpha_max=1.0,          # Maximum allowable step size
            algorithm="STRONG_WOLFE",
            m1=1e-4,                # Armijo constant (sufficient decrease)
            m2=0.1,                 # Strong-Wolfe curvature constant
        )
        
        # STEP 5: Update design variables: x(k+1) = x(k) + alpha(k) * p(k)
        # ==================================================================
        x_old = x.copy()
        x = x + alpha * p
        
        logger.debug(
            "Iteration {}: ||grad||_inf={:.3e}, "
            "||grad||_2={:.3e}, alpha={:.3e}, "
            "f(x)={:.6e}".format(iteration, grad_norm_inf, grad_norm_2, alpha, problem.objective(x))
        )
        
        iteration += 1
    
    # =========================================================================
    # CONVERGENCE FAILURE
    # =========================================================================
    
    logger.warning(
        "Steepest Descent did not converge within {} iterations. "
        "Final ||grad f|| = {:.2e}, convergence criterion = {:.2e}".format(iteration_limit, grad_norm_inf, gradient_tol)
    )
    return -1, x


if __name__ == "__main__":
    # Example: Solve Rosenbrock problem
    print("=" * 70)
    print("STEEPEST DESCENT ALGORITHM")
    print("=" * 70)
    
    # Test problem: Rosenbrock
    x0 = np.array([2.0, 2.0])
    iterations, x_final = steepest_descent(
        problem_name="rosenbrock",
        x_initial=x0,
        iteration_limit=100000,
        gradient_tol=1e-5
    )
    
    problem = get_problem("rosenbrock")
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("Problem: {}".format(problem.name))
    print("Iterations: {}".format(iterations))
    print("Final x: {}".format(x_final))
    print("Final f(x): {:.6e}".format(problem.objective(x_final)))
    if problem.minima:
        print("Expected minimum: {}".format(problem.minima[0]))
    print("=" * 70)
