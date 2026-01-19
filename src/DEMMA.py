"""
        Method of Moving Asymptotes (MMA) with Dual Extended Algorithm (DEMMA).
        
        Solves constrained optimization problems using hyperbolic approximation and 
        dual maximization. Mathematical foundation from Svanberg (1987) and 
        Slides 151-170 of Optimisation_Ch_5.pdf and Project_3_DEMMA.pdf.
        
        MATHEMATICAL FOUNDATION
        =======================
        Problem formulation:
            minimize    f(x)
            subject to  g_j(x) ≤ 0,   j=1,...,m     (m inequalities)
                        h_k(x) = 0,   k=1,...,me    (me equalities)
                        L_i ≤ x_i ≤ U_i             (variable bounds)
        
        CORE ALGORITHM COMPONENTS
        ==========================
        
        1. HYPERBOLIC APPROXIMATION (Slides 151-153)
        
           Within asymptote region [L, U], replace g, h with separable reciprocal form:
           
           For inequality constraint g_j:
               g̃_j(x) = r_j + Σ_i [ P_{ji}/(U_i - x_i) + Q_{ji}/(x_i - L_i) ]
           
           where:
               P_{ji} = (U_i - x_k_i)² · ∂g_j/∂x_i    [if ∂g_j/∂x_i > 0]
               Q_{ji} = -(x_k_i - L_i)² · ∂g_j/∂x_i   [if ∂g_j/∂x_i ≤ 0]
               r_j = g_j(x_k) - Σ_i [ P_{ji}/(U_i - x_k_i) + Q_{ji}/(x_k_i - L_i) ]
           
           For equality constraint h_k (hyperbolic symmetric form):
               h̃_k(x) = r_k + Σ_i [ P_{ki}/(U_i - x_i) - P_{ki}/(x_i - L_i) ]
           
           where:
               P_{ki} = (1/2)(U_i - x_k_i)² · ∂h_k/∂x_i
               r_k = h_k(x_k) - Σ_i [ P_{ki}/(U_i - x_k_i) - P_{ki}/(x_k_i - L_i) ]
           
           KEY PROPERTIES:
           - Separable: Each x_i decoupled → analytical minimization possible
           - Strictly convex: P > 0, Q > 0 ensures unique minimum
           - Infinitely differentiable: Smooth reciprocal approximation
        
        2. DUAL ALGORITHM (Slides 163-170)
        
           Maximize dual Lagrangian over multipliers λ, μ:
           
               L_d(λ, μ) = min_x { f̃(x) + Σ_j λ_j g̃_j(x) + Σ_k μ_k h̃_k(x) }
           
           subject to λ_j ≥ 0 (KKT condition for inequalities)
           
           EXPLICIT PRIMAL SOLUTION (Slide 162):
           From stationarity ∂L/∂x_i = 0, obtain analytical formula:
           
               x_i = (U_i√Q_i + L_i√P_i) / (√P_i + √Q_i)
           
           This x_i*(λ, μ) is optimal for any (λ, μ).
           
           DUAL GRADIENT (Slide 165):
           ∇_λ L_d = g̃(x*)  (constraint residuals)
           ∇_μ L_d = h̃(x*)  (constraint residuals)
           
           CONVERGENCE: As residuals → 0, approximate primal → exact primal
        
        3. SEARCH IN DUAL SPACE (Slides 166-168)
        
           Use Conjugate Gradient (CG) to maximize dual function over (λ, μ):
           - Search direction: Polak-Ribière formula β = g_k^T(g_k - g_{k-1}) / ||g_{k-1}||²
           - Step size: Newton-like using dual Hessian curvature
           - Multiplier updates: λ ← max(0, λ + α·S_λ), μ ← μ + α·S_μ
           - Iterate until dual residuals < tolerance
        
        KKT CONVERGENCE (Slide 164)
        ===========================
        Optimal solution satisfies necessary conditions (sufficient under LICQ):
        
        1. Stationarity: ∇L = ∇f + Σ_j λ_j ∇g_j + Σ_k μ_k ∇h_k = 0
        2. Feasibility:  g_j(x) ≤ 0, h_k(x) = 0
        3. Dual feasibility: λ_j ≥ 0
        4. Complementarity: λ_j · g_j(x) = 0 (active set condition)
        
        ILL-CONDITIONING PREVENTION
        ===========================
        
        1. COEFFICIENT FLOORS:
           P, Q bounded below by coeff_floor = 1e-15. Why:
           - Square root √P, √Q requires P, Q > 0 for real-valued output
           - Small coefficients cause 1/(U-x) to blow up (reciprocal singularity)
           - Floor ensures reciprocal stays bounded and √ remains well-defined
        
        2. ASYMPTOTE SAFEGUARDS:
           - eps = 1e-10 prevents division by zero in 1/(U-x), 1/(x-L)
           - Safe margins: x kept at least safe_margin away from L, U
           - Ensures denominators stay bounded away from zero
        
        3. MULTIPLIER BOUNDING:
           - λ_j, μ_k ∈ [-max_multiplier, max_multiplier] where max_multiplier = 100
           - Prevents Lagrange multipliers from exploding and ill-conditioning dual Hessian
           - Large multipliers amplify constraint contributions and destabilize Newton steps
        
        4. HESSIAN APPROXIMATION:
           - Diagonal approximation: H_i ≈ max(hess_floor, dy_i / ds_i) via BFGS-like update
           - Floor prevents negative/tiny diagonal elements, ensures positive definiteness
           - Captures local curvature for objective approximation convexity
        
        FEASIBILITY GUARANTEES
        ======================
        
        1. INEQUALITY CONSTRAINTS (g_j ≤ 0):
           - Dual algorithm enforces λ_j ≥ 0 at each iteration
           - Complementarity λ_j · g_j = 0 at optimum: either λ_j = 0 (constraint inactive)
             or g_j = 0 (constraint exactly satisfied)
           - Dual gradient measures constraint violation g̃(x*); as dual maximizes,
             residuals → 0, ensuring feasibility progression
        
        2. EQUALITY CONSTRAINTS (h_k = 0):
           - Hyperbolic form symmetric in ±residuals, no sign restriction on μ_k
           - Can be positive or negative to drive both h > 0 and h < 0 toward zero
           - Dual gradient measures residual |h̃(x*)|; maximization drives it to zero
        
        3. EXPLICIT PRIMAL SOLUTION ensures feasibility:
           - Every dual iterate yields analytical x* that minimizes current Lagrangian
           - This x* is interior to asymptotes (clipped to bounds + margins)
           - Feasibility improves as dual gradient (constraint residuals) decreases
        
        4. CONVERGENCE by KKT CONDITIONS:
           - Algorithm terminates when KKT satisfied: stationarity + feasibility + dual feasibility
           - Feasibility checks |g| ≤ tol, |h| ≤ tol directly
           - At termination, all constraints satisfied to specified tolerance
        
        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop MMA iterations.
        callback : callable, optional
            Called with OptimisationResult after each iteration.

Date: 2026
"""

import logging

import numpy as np
from numpy.typing import NDArray

from Convergence_LineSearch import convergence_check
from Problems import get_problem

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def demma(
    x_initial,
    lm_initial=None,
    iteration_limit=1000
):
    """
    Method of Moving Asymptotes (MMA) with Dual Extended Algorithm (DEMMA).

    Solves the chatgpt_3_modified constrained optimization problem.

    Parameters
    ----------
    x_initial : NDArray
        Initial design variables.
    lm_initial : NDArray, optional
        Initial Lagrange multipliers. If None, initialized to zero.
    iteration_limit : int, default: 1000
        Maximum number of outer-loop MMA iterations.

    Returns
    -------
    tuple
        (iteration_count, x_final, lm_final) where:
        - iteration_count: Number of iterations (-1 if limit exceeded)
        - x_final: Final design variables
        - lm_final: Final Lagrange multipliers
    """
    
    # Load problem (only chatgpt_3_modified)
    problem = get_problem("chatgpt_3_modified")
    
    n, m, me = problem.n, problem.m, problem.me
    
    x = x_initial.copy()
    if lm_initial is None:
        lm = np.zeros(m + me)
    else:
        lm = lm_initial.copy()
    
    logger.info("Starting DEMMA for problem: {}".format(problem.name))
    logger.info("Variables: {}, Inequalities: {}, Equalities: {}".format(n, m, me))
    logger.info("Initial point: {}".format(x))
    
    # =========================================================================
    # INITIALIZATION: Constants and Data Structures
    # =========================================================================
    
    # NUMERICAL SAFEGUARDS
    eps = 1e-10
    coeff_floor = 1e-15
    hess_floor = 1e-6
    max_multiplier = 100.0
    
    # CONVERGENCE TOLERANCES
    kkt_gradient_tol = 1e-5
    kkt_constraint_tol = 1e-5
    kkt_complementarity_tol = 1e-5
    
    # DUAL ALGORITHM TOLERANCE
    dual_tol = 1e-6
    dual_limit = 100
    
    # VARIABLE BOUNDS
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)
    
    # Handle infinite bounds
    lb = np.where(np.isfinite(lb), lb, -1e10)
    ub = np.where(np.isfinite(ub), ub, 1e10)
    
    # ITERATION HISTORY
    x_k = x.copy()
    x_km1 = x.copy()
    x_km2 = x.copy()
    
    # CONSTRAINT VIOLATION TRACKING
    constraint_violation = np.inf
    constraint_violation_prev = np.inf
    
    # INITIAL ASYMPTOTE WIDTHS
    delta = 0.5 * (ub - lb)
    delta = np.where(np.isfinite(delta), delta, 1.0)
    delta = np.minimum(delta, 10.0)
    
    # HESSIAN APPROXIMATION
    gradL_prev = np.zeros(n)

    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================
    
    for iteration in range(iteration_limit):
        
        # ===================================================================
        # STEP 1: EVALUATE PROBLEM AT CURRENT POINT
        # ===================================================================
        
        f_real = problem.objective(x)
        grad_f = problem.grad_objective(x)
        
        if problem.m + problem.me > 0:
            c = problem.constraints(x)
            J = problem.grad_constraints(x)
            g, h = (c[:m], c[m:]) if m > 0 else (np.zeros(0), c)
            grad_g, grad_h = (J[:m], J[m:]) if m > 0 else (np.zeros((0, n)), J)
        else:
            g, h = np.zeros(0), np.zeros(0)
            grad_g, grad_h = np.zeros((0, n)), np.zeros((0, n))

        # ===================================================================
        # STEP 2: CHECK KKT CONVERGENCE
        # ===================================================================
        
        lambda_vec, mu_vec = lm[:m], lm[m:]
        
        gradL = grad_f.copy()
        if m > 0:
            gradL += lambda_vec @ grad_g
        if me > 0:
            gradL += mu_vec @ grad_h
        
        if convergence_check(
            gradL,
            lm,
            c if (m + me > 0) else None,
            m, me,
            gradient_tol=kkt_gradient_tol,
            constraint_tol=kkt_constraint_tol,
            complementarity_tol=kkt_complementarity_tol
        ):
            logger.info("DEMMA converged at iteration {}".format(iteration))
            return iteration, x, lm

        # ===================================================================
        # STEP 3: UPDATE MOVING ASYMPTOTES
        # ===================================================================
        
        constraint_violation = 0.0
        if m > 0:
            constraint_violation += np.sum(np.maximum(g, 0.0))
        if me > 0:
            constraint_violation += np.sum(np.abs(h))
        
        if iteration > 0:
            for i in range(n):
                osc = (x_k[i] - x_km1[i]) * (x_km1[i] - x_km2[i])
                progress_ratio = max(0.01, constraint_violation_prev / max(constraint_violation, 1e-12))
                growth_factor = 1.2 + 0.1 * min(1.0, progress_ratio - 1.0)
                
                if osc > 0:
                    delta[i] *= min(1.3, growth_factor)
                elif osc < 0:
                    delta[i] *= 0.7
                
                d_range = max(1.0, ub[i] - lb[i])
                delta[i] = np.clip(delta[i], 5e-4 * d_range, 20.0 * d_range)
        
        L = np.maximum(x - delta, lb)
        U = np.minimum(x + delta, ub)

        # ===================================================================
        # STEP 4: BUILD HYPERBOLIC APPROXIMATION
        # ===================================================================
        
        P_mat = np.zeros((m + me, n))
        Q_mat = np.zeros((m + me, n))
        r_vec = np.zeros(m + me)

        # INEQUALITY CONSTRAINT APPROXIMATION
        for j in range(m):
            for i in range(n):
                df = grad_g[j, i]
                if df > 0:
                    P_mat[j, i] = (U[i] - x[i])**2 * df
                else:
                    Q_mat[j, i] = -(x[i] - L[i])**2 * df
            
            denom_p = np.maximum(U - x, eps)
            denom_q = np.maximum(x - L, eps)
            r_vec[j] = g[j] - np.sum(P_mat[j]/denom_p + Q_mat[j]/denom_q)

        # EQUALITY CONSTRAINT APPROXIMATION
        for j_loc in range(me):
            j_glob = m + j_loc
            for i in range(n):
                P_mat[j_glob, i] = 0.5 * (U[i] - x[i])**2 * grad_h[j_loc, i]
            
            term = P_mat[j_glob]
            denom_U = np.maximum(U - x, eps)
            denom_L = np.maximum(x - L, eps)
            approx = np.sum(term/denom_U - term/denom_L)
            r_vec[j_glob] = h[j_loc]

        # ===================================================================
        # STEP 5: APPROXIMATE OBJECTIVE FUNCTION
        # ===================================================================
        
        L_diag = np.ones(n)
        if iteration > 0:
            dy = gradL - gradL_prev
            ds = x - x_km1
            valid = np.abs(ds) > eps
            L_diag[valid] = np.maximum(hess_floor, dy[valid] / ds[valid])

        p0, q0 = np.zeros(n), np.zeros(n)
        
        sum_lam_p = lambda_vec @ P_mat[:m] if m > 0 else np.zeros(n)
        sum_lam_q = lambda_vec @ Q_mat[:m] if m > 0 else np.zeros(n)
        sum_mu_p = mu_vec @ P_mat[m:] if me > 0 else np.zeros(n)

        for i in range(n):
            D = delta[i]
            hess = 0.25 * D**3 * L_diag[i]
            grad = 0.5 * D**2 * gradL[i]
            
            p0[i] = max(coeff_floor, hess + grad - sum_lam_p[i] - sum_mu_p[i])
            q0[i] = max(coeff_floor, hess - grad - sum_lam_q[i] + sum_mu_p[i])

        # ===================================================================
        # STEP 6: DUAL ALGORITHM
        # ===================================================================
        
        lam_curr = np.clip(lambda_vec.copy(), 0.0, max_multiplier)
        mu_curr = np.clip(mu_vec.copy(), -max_multiplier, max_multiplier)
        
        S_vec = np.zeros(m + me)
        grad_dual_prev = np.zeros(m + me)

        for dual_iter in range(dual_limit):
            
            # --- PRIMAL MINIMIZATION ---
            
            P_tot = p0.copy()
            Q_tot = q0.copy()
            
            if m > 0:
                P_tot += lam_curr @ P_mat[:m]
                Q_tot += lam_curr @ Q_mat[:m]
            if me > 0:
                P_tot += mu_curr @ P_mat[m:]
                Q_tot -= mu_curr @ P_mat[m:]

            P_tot = np.maximum(P_tot, coeff_floor)
            Q_tot = np.maximum(Q_tot, coeff_floor)
            
            sqrt_P = np.sqrt(P_tot)
            sqrt_Q = np.sqrt(Q_tot)
            
            # EXPLICIT ANALYTICAL SOLUTION
            x_dual = (U * sqrt_Q + L * sqrt_P) / (sqrt_Q + sqrt_P)
            
            safe_margin = 0.01 * delta
            x_dual = np.clip(x_dual, lb, ub)
            x_dual = np.clip(x_dual, np.maximum(lb, L + safe_margin), 
                            np.minimum(ub, U - safe_margin))
            
            # --- DUAL GRADIENT ---
            
            grad_dual = np.zeros(m + me)
            denom_U = U - x_dual
            denom_L = x_dual - L
            
            if m > 0:
                for j in range(m):
                    grad_dual[j] = r_vec[j] + np.sum(P_mat[j]/denom_U + Q_mat[j]/denom_L)
            
            if me > 0:
                for j_loc in range(me):
                    j_glob = m + j_loc
                    grad_dual[j_glob] = r_vec[j_glob] + np.sum(P_mat[j_glob]/denom_U - P_mat[j_glob]/denom_L)
            
            # Check dual convergence
            max_dual_residual = np.max(np.abs(grad_dual)) if (m + me) > 0 else 0.0
            if max_dual_residual < dual_tol:
                break
            
            # --- CONJUGATE GRADIENT ---
            
            if dual_iter == 0:
                S_vec = grad_dual.copy()
            else:
                num = np.dot(grad_dual, grad_dual - grad_dual_prev)
                den = np.dot(grad_dual_prev, grad_dual_prev) + eps
                beta = max(0.0, num / den)
                S_vec = grad_dual + beta * S_vec

            grad_dual_prev = grad_dual.copy()
            
            # --- STEP SIZE ---
            
            denom_root = (sqrt_Q + sqrt_P)**2
            dx_dP = -0.5 * (sqrt_Q * (U - L)) / (sqrt_P * denom_root + eps)
            dx_dQ = 0.5 * (sqrt_P * (U - L)) / (sqrt_Q * denom_root + eps)
            
            S_lam = S_vec[:m]
            S_mu = S_vec[m:]
            
            vec_P_S = np.zeros(n)
            vec_Q_S = np.zeros(n)
            
            if m > 0:
                vec_P_S += S_lam @ P_mat[:m]
                vec_Q_S += S_lam @ Q_mat[:m]
            if me > 0:
                vec_P_S += S_mu @ P_mat[m:]
                vec_Q_S -= S_mu @ P_mat[m:]
                
            dx_dS = dx_dP * vec_P_S + dx_dQ * vec_Q_S
            
            dG_dx_v = np.zeros(m + me)
            inv_U2 = 1.0 / (denom_U**2 + eps)
            inv_L2 = 1.0 / (denom_L**2 + eps)
            
            if m > 0:
                term = (P_mat[:m] * inv_U2 - Q_mat[:m] * inv_L2) @ dx_dS
                dG_dx_v[:m] = term
            if me > 0:
                term = (P_mat[m:] * inv_U2 + P_mat[m:] * inv_L2) @ dx_dS
                dG_dx_v[m:] = term
                
            curvature = np.dot(S_vec, dG_dx_v)
            grad_S = np.dot(grad_dual, S_vec)
            
            if abs(curvature) < 1e-12 or grad_S < 1e-12:
                alpha = 0.1
            else:
                alpha = np.clip(-grad_S / (curvature + eps), 1e-5, 1.0)
            
            # --- MULTIPLIER UPDATE ---
            
            if m > 0:
                lam_curr = np.maximum(0.0, lam_curr + alpha * S_vec[:m])
                lam_curr = np.clip(lam_curr, 0.0, max_multiplier)
            if me > 0:
                mu_curr = mu_curr + alpha * S_vec[m:]
                mu_curr = np.clip(mu_curr, -max_multiplier, max_multiplier)

        # ===================================================================
        # STEP 7: UPDATE PRIMAL AND MULTIPLIERS
        # ===================================================================
        
        f_new = problem.objective(x_dual)
        if f_new > f_real + 1.0 or np.isnan(f_new) or not np.isfinite(f_new):
            logger.warning("Divergence detected at iteration {}".format(iteration))
            lam_curr = np.maximum(0.0, lam_curr * 0.5)
            mu_curr = mu_curr * 0.5
            x_km2 = x_km1.copy()
            x_km1 = x_k.copy()
            x_k = x.copy()
            gradL_prev = gradL.copy()
            continue
        
        x = x_dual
        if m > 0:
            lm[:m] = lam_curr
        if me > 0:
            lm[m:] = mu_curr
        
        # Update iteration history
        x_km2 = x_km1.copy()
        x_km1 = x_k.copy()
        x_k = x.copy()
        gradL_prev = gradL.copy()
        
        constraint_violation_prev = constraint_violation
        
        logger.info("Iteration {}: f={:.6e}, constraint_viol={:.3e}".format(iteration, f_new, constraint_violation))

    logger.warning("DEMMA reached iteration limit of {}".format(iteration_limit))
    return -1, x, lm


if __name__ == "__main__":
    # Example: Solve chatgpt_3_modified problem
    print("=" * 70)
    print("DEMMA (Method of Moving Asymptotes)")
    print("=" * 70)
    
    # Test problem: chatgpt_3_modified
    x0 = np.array([1.414, 1.414, 0.0, 1.0, 0.586])
    iterations, x_final, lm_final = demma(
        x_initial=x0,
        iteration_limit=1000
    )
    
    problem = get_problem("chatgpt_3_modified")
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
