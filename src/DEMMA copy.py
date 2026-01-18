    def mma(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
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
        
        Returns
        -------
        int
            Number of iterations completed at convergence, or -1 if limit reached.
        """
        
        n, m, me = self.n, self.problem.m, self.problem.me

        # =====================================================================
        # INITIALIZATION: Constants and Data Structures
        # =====================================================================
        
        # NUMERICAL SAFEGUARDS
        eps = 1e-10              # Prevents 1/(U-x) → ∞ when U ≈ x
        coeff_floor = 1e-15      # P, Q > 0 for √ operations and reciprocal stability
        hess_floor = 1e-6        # Diagonal Hessian > 0 for positive definiteness
        max_multiplier = 100.0   # Bounds on λ, μ to prevent dual ill-conditioning
        
        # CONVERGENCE TOLERANCES (fixed, not adaptive)
        kkt_gradient_tol = 1e-5       # Stationarity: |∇L| < tol
        kkt_constraint_tol = 1e-5     # Feasibility: |g|, |h| < tol
        kkt_complementarity_tol = 1e-5  # Complementarity: |λ·g| < tol
        
        # DUAL ALGORITHM TOLERANCE
        dual_tol = 1e-6          # Inner loop: stop when constraint residuals < tol
        dual_limit = 100         # Max dual iterations per outer iteration
        
        # VARIABLE BOUNDS
        lb = self.problem.lb if self.problem.lb is not None else np.full(n, -1e10)
        ub = self.problem.ub if self.problem.ub is not None else np.full(n, 1e10)

        # LAGRANGE MULTIPLIERS (warm start)
        if self.lm is None:
            self.lm = np.zeros(m + me)
        
        # ITERATION HISTORY (for asymptote updates)
        x_k = self.x.copy()      # Current iteration
        x_km1 = self.x.copy()    # Previous iteration
        x_km2 = self.x.copy()    # Two iterations back
        
        # CONSTRAINT VIOLATION TRACKING (for progress detection)
        constraint_violation = np.inf
        constraint_violation_prev = np.inf
        
        # INITIAL ASYMPTOTE WIDTHS
        delta = 0.5 * (ub - lb)
        delta = np.where(np.isfinite(delta), delta, 1.0)  # Unbounded vars: start modest
        delta = np.minimum(delta, 10.0)  # Cap for stability
        
        # HESSIAN APPROXIMATION
        gradL_prev = np.zeros(n)

        # =====================================================================
        # MAIN OPTIMIZATION LOOP
        # =====================================================================
        
        for iteration in range(iteration_limit):
            
            # ===================================================================
            # STEP 1: EVALUATE PROBLEM AT CURRENT POINT
            # ===================================================================
            
            f_real = self.problem.compute_objective(self.x)
            grad_f = self.problem.compute_grad_objective(self.x)
            
            if self.problem.constrained:
                c = self.problem.compute_constraints(self.x)
                J = self.problem.compute_grad_constraints(self.x)
                g, h = (c[:m], c[m:]) if m > 0 else (np.zeros(0), c)
                grad_g, grad_h = (J[:m], J[m:]) if m > 0 else (np.zeros((0, n)), J)
            else:
                g, h = np.zeros(0), np.zeros(0)
                grad_g, grad_h = np.zeros((0, n)), np.zeros((0, n))

            # ===================================================================
            # STEP 2: CHECK KKT CONVERGENCE (SLIDE 164)
            # ===================================================================
            
            lambda_vec, mu_vec = self.lm[:m], self.lm[m:]
            
            # Lagrangian gradient: ∇L = ∇f + λ^T ∇g + μ^T ∇h
            gradL = grad_f.copy()
            if m > 0: 
                gradL += lambda_vec @ grad_g
            if me > 0: 
                gradL += mu_vec @ grad_h
            
            # Check KKT convergence
            if self.converged(
                gradL, 
                c if self.problem.constrained else None,
                gradient_tol=kkt_gradient_tol,
                constraint_tol=kkt_constraint_tol,
                complementarity_tol=kkt_complementarity_tol
            ):
                logger.info(f"MMA converged at iteration {iteration}")
                if callback:
                    callback(OptimisationResult(iteration, self.x.copy(), objective=f_real, lm=self.lm.copy()))
                return iteration

            # ===================================================================
            # STEP 3: UPDATE MOVING ASYMPTOTES (Svanberg Heuristic, Slide 151)
            # ===================================================================
            
            # Compute current constraint violation
            constraint_violation = 0.0
            if m > 0:
                constraint_violation += np.sum(np.maximum(g, 0.0))
            if me > 0:
                constraint_violation += np.sum(np.abs(h))
            
            if iteration > 0:
                for i in range(n):
                    # Oscillation detection via cross-product
                    # osc > 0: same direction (progress)  → expand
                    # osc < 0: opposite directions (oscillation) → contract
                    osc = (x_k[i] - x_km1[i]) * (x_km1[i] - x_km2[i])
                    
                    # Adaptive expansion based on constraint progress
                    progress_ratio = max(0.01, constraint_violation_prev / max(constraint_violation, 1e-12))
                    growth_factor = 1.2 + 0.1 * min(1.0, progress_ratio - 1.0)  # 1.2-1.3
                    
                    if osc > 0:
                        delta[i] *= min(1.3, growth_factor)    # Expand 1.2-1.3x
                    elif osc < 0:
                        delta[i] *= 0.7    # Shrink 30% on oscillation
                    
                    # Keep within reasonable bounds
                    d_range = max(1.0, ub[i] - lb[i])
                    delta[i] = np.clip(delta[i], 5e-4 * d_range, 20.0 * d_range)
            
            # Asymptotes clipped to variable bounds
            L = np.maximum(self.x - delta, lb)
            U = np.minimum(self.x + delta, ub)

            # ===================================================================
            # STEP 4: BUILD HYPERBOLIC APPROXIMATION (SLIDES 151-153)
            # ===================================================================
            
            P_mat = np.zeros((m + me, n))
            Q_mat = np.zeros((m + me, n))
            r_vec = np.zeros(m + me)

            # INEQUALITY CONSTRAINT APPROXIMATION
            for j in range(m):
                for i in range(n):
                    df = grad_g[j, i]
                    # Split gradient: positive part → P, negative part → Q
                    if df > 0:
                        P_mat[j, i] = (U[i] - self.x[i])**2 * df
                    else:
                        Q_mat[j, i] = -(self.x[i] - L[i])**2 * df
                
                # Residual: r_j = g_j - approx_at_x_k
                denom_p = np.maximum(U - self.x, eps)
                denom_q = np.maximum(self.x - L, eps)
                r_vec[j] = g[j] - np.sum(P_mat[j]/denom_p + Q_mat[j]/denom_q)

            # EQUALITY CONSTRAINT APPROXIMATION (Slide 153: hyperbolic symmetric form)
            for j_loc in range(me):
                j_glob = m + j_loc
                for i in range(n):
                    # Hyperbolic: uses ±P for symmetry
                    P_mat[j_glob, i] = 0.5 * (U[i] - self.x[i])**2 * grad_h[j_loc, i]
                
                term = P_mat[j_glob]
                denom_U = np.maximum(U - self.x, eps)
                denom_L = np.maximum(self.x - L, eps)
                approx = np.sum(term/denom_U - term/denom_L)
                r_vec[j_glob] = h[j_loc]

            # ===================================================================
            # STEP 5: APPROXIMATE OBJECTIVE FUNCTION (SLIDE 162)
            # ===================================================================
            # Reciprocal form with BFGS-like Hessian diagonal
            
            # Hessian diagonal via BFGS update
            L_diag = np.ones(n)
            if iteration > 0:
                dy = gradL - gradL_prev
                ds = self.x - x_km1
                valid = np.abs(ds) > eps
                L_diag[valid] = np.maximum(hess_floor, dy[valid] / ds[valid])

            p0, q0 = np.zeros(n), np.zeros(n)
            
            # Constraint contributions
            sum_lam_p = lambda_vec @ P_mat[:m] if m > 0 else np.zeros(n)
            sum_lam_q = lambda_vec @ Q_mat[:m] if m > 0 else np.zeros(n)
            sum_mu_p = mu_vec @ P_mat[m:] if me > 0 else np.zeros(n)

            for i in range(n):
                D = delta[i]
                hess = 0.25 * D**3 * L_diag[i]
                grad = 0.5 * D**2 * gradL[i]
                
                # Ensure P > 0, Q > 0 for convexity (Slide 151)
                p0[i] = max(coeff_floor, hess + grad - sum_lam_p[i] - sum_mu_p[i])
                q0[i] = max(coeff_floor, hess - grad - sum_lam_q[i] + sum_mu_p[i])

            # ===================================================================
            # STEP 6: DUAL ALGORITHM (SLIDES 166-170)
            # ===================================================================
            # Conjugate Gradient maximization in dual space
            
            lam_curr = np.clip(lambda_vec.copy(), 0.0, max_multiplier)
            mu_curr = np.clip(mu_vec.copy(), -max_multiplier, max_multiplier)
            
            S_vec = np.zeros(m + me)          # CG search direction
            grad_dual_prev = np.zeros(m + me)

            for dual_iter in range(dual_limit):
                
                # --- PRIMAL MINIMIZATION: Explicit formula (Slide 162) ---
                # From ∂L/∂x = 0 → x_i = (U_i√Q_i + L_i√P_i) / (√P_i + √Q_i)
                
                P_tot = p0.copy()
                Q_tot = q0.copy()
                
                if m > 0:
                    P_tot += lam_curr @ P_mat[:m]
                    Q_tot += lam_curr @ Q_mat[:m]
                if me > 0:
                    P_tot += mu_curr @ P_mat[m:]
                    Q_tot -= mu_curr @ P_mat[m:]

                # Ensure positivity for √ (Slide 162)
                P_tot = np.maximum(P_tot, coeff_floor)
                Q_tot = np.maximum(Q_tot, coeff_floor)
                
                sqrt_P = np.sqrt(P_tot)
                sqrt_Q = np.sqrt(Q_tot)
                
                # EXPLICIT ANALYTICAL SOLUTION
                x_dual = (U * sqrt_Q + L * sqrt_P) / (sqrt_Q + sqrt_P)
                
                # Keep solution within bounds + safe margins
                safe_margin = 0.01 * delta
                x_dual = np.clip(x_dual, lb, ub)
                x_dual = np.clip(x_dual, np.maximum(lb, L + safe_margin), 
                                np.minimum(ub, U - safe_margin))
                
                # --- DUAL GRADIENT: Constraint residuals (Slide 165) ---
                
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
                
                # --- CONJUGATE GRADIENT: Polak-Ribière formula (Slide 166) ---
                
                if dual_iter == 0:
                    S_vec = grad_dual.copy()
                else:
                    num = np.dot(grad_dual, grad_dual - grad_dual_prev)
                    den = np.dot(grad_dual_prev, grad_dual_prev) + eps
                    beta = max(0.0, num / den)
                    S_vec = grad_dual + beta * S_vec

                grad_dual_prev = grad_dual.copy()
                
                # --- STEP SIZE: Newton-like using dual Hessian (Slide 167) ---
                
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
                
                # --- MULTIPLIER UPDATE (Slide 168) ---
                # λ ≥ 0 (KKT condition), μ unconstrained
                
                if m > 0:
                    lam_curr = np.maximum(0.0, lam_curr + alpha * S_vec[:m])
                    lam_curr = np.clip(lam_curr, 0.0, max_multiplier)
                if me > 0:
                    mu_curr = mu_curr + alpha * S_vec[m:]
                    mu_curr = np.clip(mu_curr, -max_multiplier, max_multiplier)

            # ===================================================================
            # STEP 7: UPDATE PRIMAL AND MULTIPLIERS
            # ===================================================================
            
            # DIVERGENCE CHECK: If objective got worse, don't accept step
            f_new = self.problem.compute_objective(x_dual)
            if f_new > f_real + 1.0 or np.isnan(f_new) or not np.isfinite(f_new):
                # Divergence detected: reduce multiplier magnitude and retry
                lam_curr = np.maximum(0.0, lam_curr * 0.5)
                mu_curr = mu_curr * 0.5
                # Skip update, try next iteration with reduced multipliers
                x_km2 = x_km1.copy()
                x_km1 = x_k.copy()
                x_k = self.x.copy()
                gradL_prev = gradL.copy()
                
                if callback:
                    callback(OptimisationResult(iteration, self.x.copy(), objective=f_real, lm=self.lm.copy()))
                continue
            
            self.x[:] = x_dual
            if m > 0: 
                self.lm[:m] = lam_curr
            if me > 0: 
                self.lm[m:] = mu_curr
            
            # Update iteration history
            x_km2 = x_km1.copy()
            x_km1 = x_k.copy()
            x_k = self.x.copy()
            gradL_prev = gradL.copy()
            
            if callback:
                callback(OptimisationResult(iteration, self.x.copy(), objective=f_real, lm=self.lm.copy()))
            
            # Update constraint violation tracker
            constraint_violation_prev = constraint_violation

        logger.warning(f"MMA reached iteration limit of {iteration_limit}")
        return -1