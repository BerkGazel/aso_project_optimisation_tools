"""
Problems
========

Test optimization problems with constraints and supporting mathematical functions.
Includes:
- Rosenbrock problem
- Matyas problem
- Himmelblau problem
- McCormick problem
- ChatGPT-3 Modified problem

Each problem includes objective function, constraints, and gradient calculations.

Author: ASO Project Template
Date: 2026
"""

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Problem:
    """
    Represents an optimization problem with objective and constraints.
    
    Attributes
    ----------
    name : str
        Problem name identifier.
    n : int
        Number of design variables.
    m : int
        Number of inequality constraints.
    me : int
        Number of equality constraints.
    lower_bounds : NDArray
        Lower bounds on design variables.
    upper_bounds : NDArray
        Upper bounds on design variables.
    minima : list of NDArray
        Known global minima for verification.
    """
    
    def __init__(
        self, name, n, m, me,
        objective, grad_objective,
        inequality_constraints=None, grad_inequality_constraints=None,
        equality_constraints=None, grad_equality_constraints=None,
        lower_bounds=None, upper_bounds=None, minima=None
    ):
        """Initialize a Problem instance."""
        self.name = name
        self.n = n
        self.m = m
        self.me = me
        self._objective = objective
        self._grad_objective = grad_objective
        self._inequality_constraints = inequality_constraints or []
        self._grad_inequality_constraints = grad_inequality_constraints or []
        self._equality_constraints = equality_constraints or []
        self._grad_equality_constraints = grad_equality_constraints or []
        
        if lower_bounds is not None:
            self.lower_bounds = np.asarray(lower_bounds)
        else:
            self.lower_bounds = np.full(n, -np.inf)
        
        if upper_bounds is not None:
            self.upper_bounds = np.asarray(upper_bounds)
        else:
            self.upper_bounds = np.full(n, np.inf)
        
        self.minima = minima or []
    
    def objective(self, x: NDArray) -> float:
        """Evaluate objective function f(x)."""
        return self._objective(x)
    
    def grad_objective(self, x: NDArray) -> NDArray:
        """Evaluate gradient ∇f(x)."""
        return self._grad_objective(x)
    
    def constraints(self, x: NDArray) -> NDArray:
        """Evaluate all constraints [g_1(x), ..., g_m(x), h_1(x), ..., h_me(x)]."""
        g = np.array([g(x) for g in self._inequality_constraints]) if self.m > 0 else np.array([])
        h = np.array([h(x) for h in self._equality_constraints]) if self.me > 0 else np.array([])
        return np.concatenate([g, h]) if len(g) > 0 and len(h) > 0 else (g if len(g) > 0 else h)
    
    def grad_constraints(self, x: NDArray) -> NDArray:
        """Evaluate constraint gradients (Jacobian matrix)."""
        grad_g = np.array([g(x) for g in self._grad_inequality_constraints]) if self.m > 0 else np.zeros((0, self.n))
        grad_h = np.array([h(x) for h in self._grad_equality_constraints]) if self.me > 0 else np.zeros((0, self.n))
        return np.vstack([grad_g, grad_h]) if len(grad_g) > 0 and len(grad_h) > 0 else (grad_g if len(grad_g) > 0 else grad_h)
    
    def lagrange_function(self, x: NDArray, lm: NDArray) -> float:
        """
        Evaluate Lagrange function L(x, λ, μ) = f(x) + λᵀg(x) + μᵀh(x).
        
        Parameters
        ----------
        x : NDArray
            Design variables.
        lm : NDArray
            Lagrange multipliers [lambda_1, ..., lambda_m, mu_1, ..., mu_me].
        
        Returns
        -------
        float
            Lagrange function value.
        """
        f = self._objective(x)
        
        if self.m + self.me > 0:
            c = self.constraints(x)
            lm = np.asarray(lm, dtype=float)
            for i in range(self.m + self.me):
                f += lm[i] * c[i]
        
        return f
    
    def grad_lagrange_function(self, x: NDArray, lm: NDArray) -> NDArray:
        """
        Evaluate gradient of Lagrange function ∇L(x, λ, μ) = ∇f(x) + λᵀ∇g(x) + μᵀ∇h(x).
        
        Parameters
        ----------
        x : NDArray
            Design variables.
        lm : NDArray
            Lagrange multipliers.
        
        Returns
        -------
        NDArray
            Gradient of Lagrange function.
        """
        grad = self._grad_objective(x).copy()
        
        if self.m + self.me > 0:
            J = self.grad_constraints(x)
            lm = np.asarray(lm, dtype=float)
            for i in range(self.m + self.me):
                grad += lm[i] * J[i]
        
        return grad


def get_rosenbrock_constrained() -> Problem:
    """
    Rosenbrock test problem with circular constraint.
    
    minimize    (a - x)² + b(y - x²)²
    subject to  x² + y² ≤ 1
    
    where a = 1, b = 100, x = x[0], y = x[1]
    """
    def objective(x: NDArray) -> float:
        return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2
    
    def grad_objective(x: NDArray) -> NDArray:
        return np.array([
            2.0 * (x[0] * (1.0 - 200.0 * (x[1] - x[0]**2)) - 1.0),
            200.0 * (x[1] - x[0]**2),
        ])
    
    def g1(x: NDArray) -> float:
        return x[0]**2 + x[1]**2 - 1.0
    
    def grad_g1(x: NDArray) -> NDArray:
        return np.array([2.0 * x[0], 2.0 * x[1]])
    
    return Problem(
        name="Rosenbrock (constrained)",
        n=2,
        m=1,
        me=0,
        objective=objective,
        grad_objective=grad_objective,
        inequality_constraints=[g1],
        grad_inequality_constraints=[grad_g1],
        minima=[np.array([0.78641515, 0.61769831])],
    )


def get_matyas_constrained() -> Problem:
    """
    Matyas test problem with constraint.
    
    minimize    0.26(x² + y²) - 0.48xy
    subject to  1 - x - y ≤ 0
    """
    def objective(x: NDArray) -> float:
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]
    
    def grad_objective(x: NDArray) -> NDArray:
        return np.array([
            0.52 * x[0] - 0.48 * x[1],
            0.52 * x[1] - 0.48 * x[0],
        ])
    
    def g1(x: NDArray) -> float:
        return 1.0 - x[0] - x[1]
    
    def grad_g1(x: NDArray) -> NDArray:
        return np.array([-1.0, -1.0])
    
    return Problem(
        name="Matyas (constrained)",
        n=2,
        m=1,
        me=0,
        objective=objective,
        grad_objective=grad_objective,
        inequality_constraints=[g1],
        grad_inequality_constraints=[grad_g1],
        lower_bounds=np.array([-10.0, -10.0]),
        upper_bounds=np.array([10.0, 10.0]),
        minima=[np.array([0.5, 0.5])],
    )


def get_himmelblau_constrained() -> Problem:
    """
    Himmelblau test problem with constraint.
    
    minimize    (x² + y - 11)² + (x + y² - 7)²
    subject to  y + exp(-x) ≤ 0
    """
    def objective(x: NDArray) -> float:
        return (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2
    
    def grad_objective(x: NDArray) -> NDArray:
        dx = 2.0 * (x[0]**2 + x[1] - 11.0) * 2.0 * x[0] + 2.0 * (x[0] + x[1]**2 - 7.0)
        dy = 2.0 * (x[0]**2 + x[1] - 11.0) + 2.0 * (x[0] + x[1]**2 - 7.0) * 2.0 * x[1]
        return np.array([dx, dy])
    
    def g1(x: NDArray) -> float:
        return x[1] + np.exp(-x[0])
    
    def grad_g1(x: NDArray) -> NDArray:
        return np.array([-np.exp(-x[0]), 1.0])
    
    return Problem(
        name="Himmelblau (constrained)",
        n=2,
        m=1,
        me=0,
        objective=objective,
        grad_objective=grad_objective,
        inequality_constraints=[g1],
        grad_inequality_constraints=[grad_g1],
        lower_bounds=np.array([-5.0, -5.0]),
        upper_bounds=np.array([5.0, 5.0]),
        minima=[np.array([3.584428, -1.848126])],
    )


def get_mccormick_constrained() -> Problem:
    """
    McCormick test problem with constraint.
    
    minimize    sin(x + y) + (x - y)² - 1.5x + 2.5y + 1
    subject to  (-x)³ - y - 1 ≤ 0
    """
    def objective(x: NDArray) -> float:
        return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1.0
    
    def grad_objective(x: NDArray) -> NDArray:
        dx = np.cos(x[0] + x[1]) + 2.0 * (x[0] - x[1]) - 1.5
        dy = np.cos(x[0] + x[1]) - 2.0 * (x[0] - x[1]) + 2.5
        return np.array([dx, dy])
    
    def g1(x: NDArray) -> float:
        return (-x[0])**3 - x[1] - 1.0
    
    def grad_g1(x: NDArray) -> NDArray:
        return np.array([3.0 * (-x[0])**2, -1.0])
    
    return Problem(
        name="McCormick (constrained)",
        n=2,
        m=1,
        me=0,
        objective=objective,
        grad_objective=grad_objective,
        inequality_constraints=[g1],
        grad_inequality_constraints=[grad_g1],
        lower_bounds=np.array([-1.5, -3.0]),
        upper_bounds=np.array([4.0, 4.0]),
        minima=[np.array([-0.25786071, -0.98285429])],
    )


def get_chatgpt_3_modified() -> Problem:
    """
    ChatGPT-3 Modified problem with 5 variables, 1 inequality, and 2 equality constraints.
    
    minimize    (x0 - 2)² + (x1 - 1)² + x2² + x3² + x4²
    subject to  x0² + x1² - 4 ≤ 0           (inequality)
                x2² + x3² - 1 = 0           (equality)
                x0 + x4 - 2 = 0             (equality)
    """
    def objective(x: NDArray) -> float:
        return (x[0] - 2.0)**2 + (x[1] - 1.0)**2 + x[2]**2 + x[3]**2 + x[4]**2
    
    def grad_objective(x: NDArray) -> NDArray:
        return np.array([
            2.0 * (x[0] - 2.0),
            2.0 * (x[1] - 1.0),
            2.0 * x[2],
            2.0 * x[3],
            2.0 * x[4],
        ])
    
    # Inequality constraint: g1(x) = x0² + x1² - 4 ≤ 0
    def g1(x: NDArray) -> float:
        return x[0]**2 + x[1]**2 - 4.0
    
    def grad_g1(x: NDArray) -> NDArray:
        return np.array([2.0 * x[0], 2.0 * x[1], 0.0, 0.0, 0.0])
    
    # Equality constraint: h1(x) = x2² + x3² - 1 = 0
    def h1(x: NDArray) -> float:
        return x[2]**2 + x[3]**2 - 1.0
    
    def grad_h1(x: NDArray) -> NDArray:
        return np.array([0.0, 0.0, 2.0 * x[2], 2.0 * x[3], 0.0])
    
    # Equality constraint: h2(x) = x0 + x4 - 2 = 0
    def h2(x: NDArray) -> float:
        return x[0] + x[4] - 2.0
    
    def grad_h2(x: NDArray) -> NDArray:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    
    return Problem(
        name="ChatGPT-3 Modified",
        n=5,
        m=1,
        me=2,
        objective=objective,
        grad_objective=grad_objective,
        inequality_constraints=[g1],
        grad_inequality_constraints=[grad_g1],
        equality_constraints=[h1, h2],
        grad_equality_constraints=[grad_h1, grad_h2],
        minima=[np.array([1.81813691, 0.83329358, 0.0, 1.0, 0.18186309])],
    )


def get_problem(problem_name: str) -> Problem:
    """
    Retrieve a test problem by name.
    
    Parameters
    ----------
    problem_name : str
        One of: "rosenbrock", "matyas", "himmelblau", "mccormick", "chatgpt_3_modified"
    
    Returns
    -------
    Problem
        The requested optimization problem.
    
    Raises
    ------
    ValueError
        If problem_name is not recognized.
    """
    problems = {
        "rosenbrock": get_rosenbrock_constrained,
        "matyas": get_matyas_constrained,
        "himmelblau": get_himmelblau_constrained,
        "mccormick": get_mccormick_constrained,
        "chatgpt_3_modified": get_chatgpt_3_modified,
    }
    
    if problem_name not in problems:
        raise ValueError("Unknown problem: {}. Available: {}".format(problem_name, list(problems.keys())))
    
    return problems[problem_name]()
