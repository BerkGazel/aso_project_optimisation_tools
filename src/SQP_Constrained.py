"""
        Sequential Quadratic Programming (SQP) with BFGS Quasi-Newton Method.
        
        Solves constrained nonlinear optimization problems:
        
            minimize    f(x)
            subject to  g_j(x) ≤ 0,   j = 1, ..., m        (inequality constraints)
                        h_k(x) = 0,   k = 1, ..., m_e      (equality constraints)
        
        ALGORITHM OUTLINE
        =================
        At each iteration k:
        
        1. Evaluate objective f, constraints g, h at current point x_k
        2. Check KKT convergence conditions
        3. Identify active constraints
        4. Solve QP subproblem with BFGS-approximated Hessian
        5. Find search direction p from QP solution
        6. Perform line search to find step size α
        7. Update primal variables: x_{k+1} = x_k + α·p
        8. Update Lagrange multipliers λ
        9. Update Hessian approximation H using BFGS formula
        
        MATHEMATICAL FOUNDATION
        =======================
        
        KKT Optimality Conditions (for convergence check):
          - Stationarity:      ∇L(x,λ) = 0  where L = f + λᵀg + μᵀh
          - Primal feasibility: g(x) ≤ 0,  h(x) = 0
          - Dual feasibility:  λ ≥ 0
          - Complementarity:   λ_j·g_j(x) = 0
        
        QP Subproblem (quadratic model of Lagrangian):
          minimize    p^T·∇f + (1/2)·p^T·H·p
          subject to  ∇g_j^T·p + g_j(x) ≤ 0,   j ∈ active
                      ∇h_k^T·p + h_k(x) = 0,   k = 1,...,m_e
        
        The KKT system for the QP subproblem is:
          [ H    J^T ] [ p  ] = [ -∇f ]
          [ J     0  ] [ λ' ] = [ -c  ]
        
        where H is the BFGS-approximated Hessian of the Lagrangian,
        J contains gradients of active constraints.
        
        BFGS Update (rank-2 symmetric update):
          H_new = (I - ρ·y·s^T)·H·(I - ρ·s·y^T) + ρ·y·y^T
          
          where:
            s = x_new - x_old         (primal step)
            y = ∇L_new - ∇L_old       (curvature estimate)
            ρ = 1 / (s^T·y)

        References
        ----------
        .. [1] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
               Springer. Chapters 12-18.
        .. [2] Optimisation_Ch_4.pdf - SQP algorithms and Quasi-Newton methods.
        
Date: 2026
"""

import logging

import numpy as np
from numpy.typing import NDArray

from Convergence_LineSearch import convergence_check, line_search
from Problems import get_problem

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def sqp_constrained(
    problem_name, x_initial,
    lm_initial=None, iteration_limit=1000
):
    """
    Sequential Quadratic Programming (SQP) with BFGS Quasi-Newton Method.
    
    Parameters
    ----------
    problem_name : str
        Name of the constrained problem to solve.
    x_initial : NDArray
        Initial design variables.
    lm_initial : NDArray, optional
        Initial Lagrange multipliers. If None, initialized to zero.
    iteration_limit : int, default: 1000
        Maximum number of outer-loop iterations.

    Returns
    -------
    tuple
        (iteration_count, x_final, lm_final) where:
        - iteration_count: Number of iterations (-1 if limit exceeded)
        - x_final: Final design variables
        - lm_final: Final Lagrange multipliers
    """
    
    # Load problem
    problem = get_problem(problem_name)
    
    # Initialize design variables and Lagrange multipliers
    x = x_initial.copy()
    n = x.size
    m = problem.m
    me = problem.me
    
    if lm_initial is None:
        lm = np.zeros(m + me)
    else:
        lm = lm_initial.copy()
    
    logger.info("Starting SQP for problem: {}".format(problem.name))
    logger.info("Variables: {}, Inequalities: {}, Equalities: {}".format(n, m, me))
    logger.info("Initial point: {}".format(x))
    
    # =========================================================================
    # INITIALIZATION: Algorithm parameters
    # =========================================================================
    
    # KKT CONVERGENCE TOLERANCES
    kkt_grad_tol = 1e-3
    kkt_const_tol = 1e-3
    kkt_compl_tol = 1e-3
    
    # ACTIVE CONSTRAINT DETECTION
    tol_active = 1e-4
    tol_lm = 1e-7
    
    # BFGS SAFEGUARDS
    tol_curv = 1e-8
    
    # LINE SEARCH PARAMETERS
    max_backtracks = 25
    rho_merit = 100.0
    
    # =========================================================================
    # DATA STRUCTURES
    # =========================================================================
    
    # Hessian approximation H (initialized as identity)
    H = np.eye(n)
    first_bfgs_done = False
    
    # =========================================================================
    # MAIN SQP LOOP
    # =========================================================================
    
    for iteration in range(iteration_limit):
        
        # =================================================================
        # STEP 1: EVALUATE PROBLEM AT CURRENT POINT
        # =================================================================
        
        x_k = x.copy()
        lambda_k = lm.copy()
        
        f_k = problem.objective(x_k)
        
        if m + me > 0:
            c_k = problem.constraints(x_k)
            J_k = problem.grad_constraints(x_k)
        else:
            c_k = np.zeros(0)
            J_k = np.zeros((0, n))
        
        gradL_k = problem.grad_lagrange_function(x_k, lambda_k)
        
        logger.info("Iteration {}: f={:.6e}, max|∇L|={:.3e}".format(iteration, f_k, np.max(np.abs(gradL_k))))
        
        # =================================================================
        # STEP 2: CHECK CONVERGENCE USING KKT CONDITIONS
        # =================================================================
        
        if convergence_check(
            gradient=gradL_k,
            lagrange_multipliers=lm,
            constraints=c_k if (m + me > 0) else None,
            problem_m=m,
            problem_me=me,
            gradient_tol=kkt_grad_tol,
            constraint_tol=kkt_const_tol,
            complementarity_tol=kkt_compl_tol
        ):
            logger.info("SQP converged after {} iterations".format(iteration))
            return iteration, x, lm
        
        # =================================================================
        # STEP 3: BUILD ACTIVE CONSTRAINT SET
        # =================================================================
        
        active_idx = []
        
        if m > 0:
            for j in range(m):
                if c_k[j] >= -tol_active or lambda_k[j] > tol_lm:
                    active_idx.append(j)
        
        if me > 0:
            active_idx.extend(range(m, m + me))
        
        m_a = len(active_idx)
        if m_a > 0:
            J_a = J_k[active_idx, :]
            c_a = c_k[active_idx]
        else:
            J_a = np.zeros((0, n))
            c_a = np.zeros(0)
        
        logger.debug("  Active constraints: {} out of {}".format(m_a, m + me))
        
        # =================================================================
        # STEP 4: SOLVE QP SUBPROBLEM
        # =================================================================
        
        if m_a == 0:
            try:
                p = -np.linalg.solve(H, gradL_k)
                delta_lambda_a = np.zeros(0)
            except np.linalg.LinAlgError:
                logger.warning("  Hessian singular (no active). Using steepest descent.")
                p = -gradL_k.copy()
                H = np.eye(n)
                delta_lambda_a = np.zeros(0)
        
        else:
            dim_kkt = n + m_a
            KKT = np.zeros((dim_kkt, dim_kkt))
            
            KKT[:n, :n] = H
            KKT[:n, n:] = J_a.T
            KKT[n:, :n] = J_a
            
            rhs = np.zeros(dim_kkt)
            rhs[:n] = -gradL_k
            rhs[n:] = -c_a
            
            try:
                sol = np.linalg.solve(KKT, rhs)
                p = sol[:n]
                delta_lambda_a = sol[n:]
            except np.linalg.LinAlgError:
                logger.warning("  KKT system singular. Using steepest descent.")
                H = np.eye(n)
                p = -gradL_k.copy()
                delta_lambda_a = np.zeros(m_a)
        
        # =================================================================
        # STEP 5: MAP ACTIVE SET MULTIPLIERS BACK TO FULL VECTOR
        # =================================================================
        
        delta_lambda_full = np.zeros(m + me)
        for local_idx, global_idx in enumerate(active_idx):
            delta_lambda_full[global_idx] = delta_lambda_a[local_idx]
        
        lambda_candidate = lambda_k + delta_lambda_full
        
        if m > 0:
            lambda_candidate[:m] = np.maximum(0.0, lambda_candidate[:m])
        
        # =================================================================
        # STEP 6: VERIFY SEARCH DIRECTION IS DESCENT
        # =================================================================
        
        directional_deriv = float(np.dot(gradL_k, p))
        
        if directional_deriv >= 0:
            logger.warning(
                "  Search direction not descent (∇L^T p={:.3e}). "
                "Using steepest descent.".format(directional_deriv)
            )
            p = -gradL_k.copy()
            H = np.eye(n)
        
        # =================================================================
        # STEP 7: LINE SEARCH TO FIND STEP SIZE
        # =================================================================
        
        alpha = line_search(
            objective_func=problem.objective,
            grad_objective_func=problem.grad_objective,
            x_current=x,
            direction=p,
            alpha_ini=1.0,
            alpha_min=1e-8,
            alpha_max=1000.0,
            algorithm="STRONG_WOLFE",
            m1=1e-4,
            m2=0.9,
        )
        
        # ===================================================================
        # MERIT FUNCTION BACKTRACKING
        # ===================================================================
        
        def compute_merit(x_val: NDArray, rho: float) -> float:
            """Compute L1 penalty merit function."""
            f_val = problem.objective(x_val)
            
            if m + me == 0:
                return f_val
            
            c_val = problem.constraints(x_val)
            
            g_viol = np.sum(np.maximum(0.0, c_val[:m])) if m > 0 else 0.0
            h_viol = np.sum(np.abs(c_val[m:])) if me > 0 else 0.0
            
            return f_val + rho * (g_viol + h_viol)
        
        merit_k = compute_merit(x_k, rho_merit)
        
        alpha_accepted = alpha
        n_backtracks = 0
        
        while n_backtracks < max_backtracks:
            x_trial = x_k + alpha_accepted * p
            merit_trial = compute_merit(x_trial, rho_merit)
            
            grad_f_k = problem.grad_objective(x_k)
            grad_deriv = float(np.dot(grad_f_k, p))
            armijo_rhs = merit_k + 1e-4 * alpha_accepted * grad_deriv
            
            if merit_trial <= armijo_rhs:
                break
            
            alpha_accepted *= 0.5
            n_backtracks += 1
            
            if alpha_accepted < 1e-8:
                rho_merit *= 10.0
                merit_k = compute_merit(x_k, rho_merit)
                logger.debug("    Small step → increased penalty to {:.1e}".format(rho_merit))
        
        logger.debug("  Line search: α={:.3e}, backtracks={}".format(alpha_accepted, n_backtracks))
        
        # Check for stagnation
        if alpha_accepted < 1e-9 and np.linalg.norm(p) < 1e-10:
            logger.info("SQP converged due to stagnation (step size {:.3e})".format(alpha_accepted))
            return iteration, x, lm
        
        # ===================================================================
        # STEP 8: UPDATE PRIMAL AND DUAL VARIABLES
        # ===================================================================
        
        s_k = alpha_accepted * p
        x = x_k + s_k
        lm = lambda_candidate
        
        logger.debug("  Step norm: {:.3e}".format(np.linalg.norm(s_k)))
        
        # Early convergence check
        if np.linalg.norm(s_k) < 1e-12:
            logger.info("SQP converged after {} iterations (small step norm)".format(iteration))
            return iteration, x, lm
        
        # =================================================================
        # STEP 9: UPDATE HESSIAN APPROXIMATION VIA BFGS
        # =================================================================
        
        gradL_kp1 = problem.grad_lagrange_function(x, lm)
        gradL_old_new_lm = problem.grad_lagrange_function(x_k, lm)
        
        y_k = gradL_kp1 - gradL_old_new_lm
        
        sy = float(np.dot(s_k, y_k))
        Hs = H @ s_k
        sHs = float(np.dot(s_k, Hs))
        
        logger.debug("  BFGS: s^T y={:.3e}, s^T H s={:.3e}".format(sy, sHs))
        
        # POWELL DAMPING
        if sy <= 0.0 or sy < 0.2 * sHs:
            if abs(sHs - sy) > 1e-20:
                theta = (0.8 * sHs) / (sHs - sy)
                theta = np.clip(theta, 0.0, 1.0)
            else:
                theta = 1.0
            
            y_tilde = theta * y_k + (1.0 - theta) * Hs
            logger.debug(f"    Powell damping: θ={theta:.3f}")
        else:
            y_tilde = y_k
        
        sy_tilde = float(np.dot(s_k, y_tilde))
        
        # =================================================================
        # BFGS UPDATE
        # =================================================================
        
        if sy_tilde > tol_curv:
            
            if not first_bfgs_done:
                yy = float(np.dot(y_tilde, y_tilde))
                if yy > 1e-12:
                    gamma = sy_tilde / yy
                    gamma = np.clip(gamma, 1e-3, 1e3)
                    H = gamma * np.eye(n)
                    first_bfgs_done = True
                    logger.debug("    Initial Hessian scaling: γ={:.3e}".format(gamma))
                    Hs = H @ s_k
                    sHs = float(np.dot(s_k, Hs))
            
            rho = 1.0 / sy_tilde
            I = np.eye(n)
            
            term1 = I - rho * np.outer(y_tilde, s_k)
            term2 = I - rho * np.outer(s_k, y_tilde)
            H = term1 @ H @ term2 + rho * np.outer(y_tilde, y_tilde)
            
            logger.debug("    BFGS update applied")
        else:
            logger.debug("    BFGS update skipped (curvature={:.3e} < tol)".format(sy_tilde))
    
    # Iteration limit reached
    logger.warning("SQP algorithm reached iteration limit ({}) without convergence".format(iteration_limit))
    return -1, x, lm


if __name__ == "__main__":
    # Example: Solve constrained Rosenbrock problem
    print("=" * 70)
    print("SQP CONSTRAINED OPTIMIZATION")
    print("=" * 70)
    
    # Test problem: Rosenbrock with constraint
    x0 = np.array([2.0, 2.0])
    iterations, x_final, lm_final = sqp_constrained(
        problem_name="rosenbrock",
        x_initial=x0,
        iteration_limit=1000
    )
    
    problem = get_problem("rosenbrock")
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("Problem: {}".format(problem.name))
    print("Iterations: {}".format(iterations))
    print("Final x: {}".format(x_final))
    print("Final f(x): {:.6e}".format(problem.objective(x_final)))
    print("Final Lagrange multipliers: {}".format(lm_final))
    if problem.minima:
        print("Expected minimum: {}".format(problem.minima[0]))
    print("=" * 70)
