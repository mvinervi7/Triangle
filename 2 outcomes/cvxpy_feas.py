from typing import Literal, Dict
from inflation import InflationProblem, InflationLP
from inflation.lp.monomial_classes import CompoundMoment
import numpy as np
import cvxpy as cp
import sympy as sp

from scipy.optimize import bisect


class CVXPY_feasibility_as_optimisation:
    def __init__(self, inflation_scenario_program) -> None:
        from tqdm import tqdm
        from scipy.sparse import coo_matrix

        self.inflation_scenario_program = inflation_scenario_program
        self.known_variables = set(
            str(m) for m in self.inflation_scenario_program.monomials if m.knowability_status == 'Knowable')
        self.free_variables = set(str(m) for m in self.inflation_scenario_program.monomials).difference(
            self.known_variables)
        assert len(inflation_scenario_program.monomials) == len(self.known_variables) + len(
            self.free_variables), "Some variables are missing"

        args = inflation_scenario_program._prepare_solver_arguments()
        objective, inequalities, equalities, semiknown_vars, known_vars \
            = args['objective'], args['inequalities'], args['equalities'], args['semiknown_vars'], args['known_vars']
        assert len(equalities) == 0, "Manual CVXPY for equalities currently not supported"

        self.lam = cp.Variable(name='lam')

        self.x = cp.Variable(len(self.free_variables), name='x')
        self.x_known = cp.Parameter(len(self.known_variables), name='x_known')

        # Check if 'constant_term' is in any expression
        constant_term = False
        _eq_vars = set()
        for eq in equalities:
            _eq_vars.update(eq.keys())
        if 'constant_term' in _eq_vars:
            constant_term = True
        _ineq_vars = set()
        for ineq in inequalities:
            _ineq_vars.update(ineq.keys())
        if 'constant_term' in _ineq_vars:
            constant_term = True
        if 'constant_term' in objective:
            constant_term = True

        if constant_term:
            print("WARNING: constant_term is in the objective or inequalities or equalities, this is not supported")
        else:
            known_vars.pop('constant_term')

        self.x_known_index = {x: i for i, x in enumerate(self.known_variables)}
        self.x_index = {x: i for i, x in enumerate(self.free_variables)}

        Arow, Acol, Adata = [], [], []
        Brow, Bcol, Bdata = [], [], []
        for i, ineq in tqdm(enumerate(inequalities), desc="Extracting A, B such that A @ x + B @ x_known >= 0"):
            _x_ = set(ineq.keys())
            _x_free = _x_.intersection(self.free_variables)
            _x_known = _x_.intersection(self.known_variables)
            for x in _x_free:
                Arow.append(i)
                Acol.append(self.x_index[x])
                Adata.append(ineq[x])
            for x in _x_known:
                Brow.append(i)
                Bcol.append(self.x_known_index[x])
                Bdata.append(ineq[x])

        self.A = coo_matrix((Adata, (Arow, Acol)), shape=(len(inequalities),
                                                          len(self.free_variables)))
        self.B = coo_matrix((Bdata, (Brow, Bcol)), shape=(len(inequalities),
                                                          len(self.known_variables)))

        # Instead of Ax+b >= 0, x>=0 we solve the relaxed problem Ax+b >= lambda · 1_c, x>=lambda · 1_v, lambda >= 0
        # where if max lambda is positive, the original is feasible, if max lambda is negative, the original is infeasible
        self.constraints = [self.A @ self.x + self.B @ self.x_known + self.lam * (-1 * np.ones(self.A.shape[0])) >= 0]
        self.constraints += [self.x + self.lam * (-1 * np.ones(self.A.shape[1])) >= 0]
        self.cvxpy_problem = cp.Problem(cp.Maximize(self.lam), self.constraints)

    def set_p(self, prob):
        if isinstance(prob, dict):
            as_vector = [0] * len(self.known_variables)
            for k, v in prob.items():
                as_vector[self.x_known_index[k]] = v
            self.x_known.value = as_vector
        elif isinstance(prob, np.ndarray):
            self.inflation_scenario_program.set_distribution(prob)
            _dict1 = {str(k): v for k, v in self.inflation_scenario_program.known_moments.items()}
            as_vector = [0] * len(self.known_variables)
            for x in self.known_variables:
                as_vector[self.x_known_index[x]] = _dict1[x]
            self.x_known.value = as_vector
        else:
            raise ValueError("dictCG_p1 must be of type dict or both of type np.ndarray")

    def solve(self, prob=None, verbose=False):
        if prob is not None:
            self.set_p(prob)
        try:
            self.cvxpy_problem.solve(verbose=verbose, solver=cp.MOSEK)
            vv = max(self.lam.value, self.cvxpy_problem.solution.opt_val)

            cert_coeffs = self.B.T @ self.constraints[0].dual_value
            cert_as_dict = {self.inflation_scenario_program.monomial_from_name[x]: cert_coeffs[self.x_known_index[x]]
                            for x in self.known_variables}
            cert_as_probs = sum([sp.Symbol(str(x)) * v for x, v in cert_as_dict.items()]).subs({sp.Symbol('1'): 1})
            self.solution = vv, cert_as_probs, cert_as_dict
        except cp.error.SolverError:
            self.solution = np.inf, -1, {'1': 1}
        return self.solution