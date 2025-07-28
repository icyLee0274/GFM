from typing import Any

import numpy as np
import pickle
import cvxpy as cp
import sympy as sp
import pyomo.environ as pyo
import multiprocessing as mp
import scipy.io
import os
import itertools
import networkx as nx
from numpy import ndarray, dtype, floating
from sympy.polys import monomials

n_process = 50


def generate_positive_definite_matrix(
        d: int,
        low: float = 1e-6,
        high: float = 1.0,
        seed: int = None
) -> ndarray:
    """
    Generates a random positive definite matrix of size d x d.

    The matrix is generated as

    .. math::

        M = Q D Q^T,

    where :math:`Q` is a random orthogonal matrix and :math:`D` is a diagonal matrix with positive eigenvalues sampled uniformly.

    :param d: Dimension of generated matrix.
    :param low: Lower bound for eigenvalues, must be positive (non-negative).
    :param high: Upper bound for eigenvalues.
    :param seed: RNG seed.
    :return: A numpy array of shape (d, d) representing a positive definite matrix.
    """
    rng = np.random.default_rng(seed)

    ev = rng.uniform(low=low, high=high, size=d)
    D = np.diag(ev)

    Z = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(Z)

    M = Q @ D @ Q.T

    return M


def generate_monomials(variables, degree, constant: bool = False):
    """
    Generates a list of all monomials for a given set of variables and degree.

    :param variables: A list of :code:`sympy.Symbol`, e.g., [x1, x2].
    :param degree: Maximum degree of the monomials to generate.
    :param constant: If ``True``, include the constant term ``1`` in the list of monomials.

    :return: List of sympy expressions: A sorted list of monomials.
    """
    monomials = [sp.Integer(1)] if constant else []
    for d in range(1, degree + 1):
        # Generate all combinations of variables with repetition
        for p in itertools.combinations_with_replacement(variables, d):
            monomials.append(np.prod(p))

    # Remove duplicates and sort
    return sorted(list(set(monomials)), key=str)


def generates_sos_polynomial(n_var, half_degree):
    """
    Generates a sum-of-squares polynomial of given degree in n_var variables.

    The polynomial is generated as a sum of squares of monomials:

    .. math::

        p(x) = z(x)^T Q z(x),

    where :math:`z(x)` is a vector of monomials of degree :math:`d` and :math:`Q` is a positive definite matrix.

    :param n_var: Number of variables.
    :param half_degree: Half of degree of the polynomial.
    :return: A sympy expression representing the SOS polynomial.
    """

    # Generate symbolic variables, [x_1, ..., x_n]
    variables = sp.symbols(f'x1:{n_var + 1}')

    # Generate monomials
    monos = generate_monomials(variables, half_degree, constant=False)

    # Create a random polynomial with coefficients
    Q = generate_positive_definite_matrix(len(monos))

    n_mono = len(monos)
    poly = sum(float(Q[i, j]) * monos[i] * monos[j] for i, j in itertools.product(range(n_mono), repeat=2))

    return poly


def generate_opt(
        defaults,
        opt_problem,
        instance_para_list,
        input_bound=None,
        output_bound=None,
        parallel=True,
        solver='gurobi'
) -> Any:
    print(opt_problem, instance_para_list)
    if not os.path.exists('datasets/{}'.format(opt_problem)):
        os.makedirs('datasets/{}'.format(opt_problem))
    seed = defaults['seed']
    if opt_problem == 'qp':
        num_var = instance_para_list[0]
        num_ineq = instance_para_list[1]
        num_eq = instance_para_list[2]
        test_size = instance_para_list[3]
        data = make_qp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver)
    elif opt_problem == 'socp':
        num_var = instance_para_list[0]
        num_ineq = instance_para_list[1]
        num_eq = instance_para_list[2]
        test_size = instance_para_list[3]
        data = make_socp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver)
    elif opt_problem == 'convex_qcqp':
        num_var = instance_para_list[0]
        num_ineq = instance_para_list[1]
        num_eq = instance_para_list[2]
        test_size = instance_para_list[3]
        data = make_convex_qcqp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver)
    elif opt_problem == 'sdp':
        num_var = instance_para_list[0]
        num_ineq = instance_para_list[1]
        num_eq = instance_para_list[2]
        test_size = instance_para_list[3]
        data = make_sdp(seed, int(num_var ** 0.5), num_ineq, num_eq, test_size, input_bound, output_bound, parallel,
                        solver)
    elif opt_problem == 'jccim':
        num_var = instance_para_list[0]
        num_ineq = instance_para_list[1]
        num_eq = instance_para_list[2]
        test_size = instance_para_list[3]
        num_scenario = instance_para_list[4]
        data = make_jcc_im(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver,
                           num_scenario)
    elif opt_problem == 'graph_qp':
        num_var = instance_para_list[0]
        num_ineq = instance_para_list[1]
        num_eq = instance_para_list[2]
        test_size = instance_para_list[3]
        num_node = instance_para_list[4]
        sparsity = instance_para_list[5]
        fix_graph = instance_para_list[6]
        data = make_graph_qp(seed, num_node, sparsity, fix_graph, num_var, num_ineq, num_eq, test_size, input_bound,
                             output_bound, parallel,
                             solver)
    elif opt_problem == 'power_control':
        num_node = instance_para_list[0]
        test_size = instance_para_list[1]
        data = make_power_control(seed, num_node, test_size, parallel, solver)
    else:
        NotImplementedError('Should never reach here.')

    if opt_problem in ['qp', 'socp', 'convex_qcqp', 'sdp']:
        with open("datasets/{}/random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".
                          format(opt_problem, seed, opt_problem, num_var, num_ineq, num_eq, test_size), 'wb') as f:
            pickle.dump(data, f)
    elif opt_problem == 'jccim':
        num_scenario = instance_para_list[4]
        with open("datasets/{}/random_{}_{}_dataset_var{}_ineq{}_eq{}_scenario{}_ex{}".
                          format(opt_problem, seed, opt_problem, num_var, num_ineq, num_eq, num_scenario, test_size),
                  'wb') as f:
            pickle.dump(data, f)
    elif 'graph' in opt_problem:
        with open("datasets/{}/random_{}_{}_dataset_var{}_ineq{}_eq{}_node{}_spar{}_fix{}_ex{}".
                          format(opt_problem, seed, opt_problem, num_var, num_ineq, num_eq, num_node, sparsity,
                                 fix_graph, test_size), 'wb') as f:
            pickle.dump(data, f)
    elif opt_problem == 'power_control':
        with open("datasets/{}/random_{}_{}_dataset_node{}_ex{}".
                          format(opt_problem, seed, opt_problem, num_node, test_size), 'wb') as f:
            pickle.dump(data, f)
    else:
        NotImplementedError('Seem unreachable')

    return data


def find_partial_variable(A, num_var, num_eq):
    i = 0
    det_min = 0
    best_partial = 0
    while i < 1000:
        # np.random.seed(i)
        partial_vars_idx = np.random.choice(num_var, num_var - num_eq, replace=False)
        other_vars = np.setdiff1d(np.arange(num_var), partial_vars_idx)
        _, det = np.linalg.slogdet(A[:, other_vars])
        if det > det_min:
            det_min = det
            print('best_det', det_min, end='\r')
            best_partial = partial_vars_idx
        i += 1
    print('best_det', det_min)
    return best_partial


def make_qp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver):
    # min 1/2 y^TQy + p^Ty
    # s.t Ay == X_i
    #     Gy <= h
    np.random.seed(seed)
    XL, XU = input_bound
    YL, YU = output_bound
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
    data = {'Q': Q, 'p': p, 'A': A, 'X': X, 'G': G, 'h': h, 'XL': XL, 'XU': XU, 'YL': YL, 'YU': YU, }
    if parallel:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, p, G, h, YU, YL, A, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_qp, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_qp((i, Xi, num_var, Q, p, G, h, YU, YL, A, solver))
            if yt is not None: Y.append(yt)
    data['Y'] = np.array(Y)
    data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data


def solve_qp(args):
    n, Xi, num_var, Q, p, G, h, YU, YL, A, solver = args
    y = cp.Variable(num_var)
    constraints = [G @ y <= h, y <= YU, y >= YL, A @ y == Xi]
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y), constraints)
    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")

    return y.value


def make_soc(
        num_var: int,
        num_ineq: int,
        seed: int | None = None,
        solver: str = 'mosek',
        n_retry: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate second-order cone constraints
    \[
        || Ax + b ||_2 <= c^Tx + d.
    \]
    :param num_var: Number of variables, i.e., dimension.
    :param num_ineq: Number of inequalities.
    :param seed: Random seed.
    :param solver: String of cvxpy solver.
    :param n_retry: Maximum number of retry attempts.
    :return: A tuple of 4 numpy arrays, one for each variable.
    """
    rng = np.random.default_rng(seed)

    As = rng.normal(size=(num_ineq, num_var, num_var))
    bs = rng.normal(size=(num_ineq, num_var))
    cs = rng.normal(size=(num_ineq, num_var))
    xs = rng.normal(size=(num_ineq, num_var))
    ds = (np.linalg.norm((As @ xs[:, :, np.newaxis]).squeeze() + bs, axis=1)
          - np.sum(cs * xs, axis=1)
          + rng.uniform(size=num_ineq))

    return As, bs, cs, ds.flatten()


def make_qc(
        num_var: int,
        num_ineq: int,
        seed: int | None = None,
        n_retry: int = 1000,
        solver: str = 'mosek',
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Generate quadratic constraints
    \[
        x^T Q x + p^T x + b <= 0
    \]

    :param num_var: Number of variables, i.e., dimension.
    :param num_ineq: Number of inequalities.
    :param seed: Random seed.
    :param n_retry: Maximum number of retry attempts.
    :param solver: String of cvxpy solver.
    :return: A tuple of three numpy arrays, the first dimension is of `num_ineq`.
    """
    rng = np.random.default_rng(seed)

    # Generate a random matrix A and compute Q = A^TA to ensure PSD
    As = rng.standard_normal((num_ineq, num_var, num_var))
    Qs = np.einsum("bik, bjk -> bij", As, As)

    # Generate random linear term and scalar constant
    ps = rng.normal(size=(num_ineq, num_var))
    bs = rng.uniform(-2, 0, size=[num_ineq])

    return Qs, ps, bs


def make_polytope(
        seed: int | None = None,
        num_var: int = 2,
        num_ineq: int = 4,
        test_size: int = 10,
        bound: tuple[float, float] = (-1, 1),
        solver: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Generate polytope defined as
    \[
        Gx <= h
    \]

    :param seed: Random seed.
    :param num_var: Number of variables.
    :param num_ineq: Number of inequalities.
    :param test_size: Maximum retry times.
    :param bound: Upper and lower bounds for input.
    :param solver: String of cxvpy solvers, None for default solver.

    :return: Tuple of G, h, and a (feasible) point.
    """
    lower, upper = bound
    rng = np.random.default_rng(seed)

    y = cp.Variable(num_var)
    objective = cp.Minimize(1)

    for i in range(test_size):
        G = rng.uniform(low=-1, high=1, size=[num_ineq, num_var])
        h = np.sum(np.abs(G @ rng.uniform(-1, 1, [num_var, num_ineq])), axis=1)
        constraints = [y >= lower, y <= upper, G @ y <= h]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)
        if problem.status == cp.OPTIMAL:
            return G, h, y.value
    return None


def make_socp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver):
    np.random.seed(seed)
    """input-output para"""
    XL = input_bound[0]
    XU = input_bound[1]
    YL = output_bound[0]
    YU = output_bound[1]
    """Obj para"""
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)
    """Eq para"""
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    """Ineq para"""
    x0 = np.random.uniform(-1, 1, size=(num_var,))
    G = np.random.uniform(-1, 1, size=(num_ineq, num_ineq, num_var))
    h = np.random.uniform(-1, 1, size=(num_ineq, num_ineq))
    C = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    d = np.linalg.norm(G @ x0 + h, ord=2, axis=1) - C @ x0
    """data set"""
    data = {'Q': Q, 'p': p,
            'A': A, 'X': X,
            'G': np.array(G), 'h': np.array(h),
            'C': np.array(C), 'd': np.array(d),
            'XL': XL, 'XU': XU,
            'YL': YL, 'YU': YU,
            'Y': []}
    if parallel:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, p, G, h, C, d, YU, YL, A, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_socp, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_socp((i, Xi, num_var, Q, p, G, h, C, d, YU, YL, A, solver))
            if yt is not None: Y.append(yt)

    data['Y'] = np.array(Y)
    data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data


def solve_socp(args):
    n, Xi, num_var, Q, p, G, h, C, d, YU, YL, A, solver = args
    y = cp.Variable(num_var)
    soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(C.shape[0])]
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                      soc_constraints + [A @ y == Xi, y <= YU, y >= YL])
    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")

    return y.value


def make_convex_qcqp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver):
    np.random.seed(seed)
    """input-output para"""
    XL = input_bound[0]
    XU = input_bound[1]
    YL = output_bound[0]
    YU = output_bound[1]
    """Obj para"""
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)
    """Eq para"""
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    """Ineq para"""
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
    H = np.random.uniform(0, 0.1, size=(num_ineq, num_var))
    H = [np.diag(H[i]) for i in range(num_ineq)]
    H = np.array(H)
    """data set"""
    data = {'Q': Q, 'p': p,
            'A': A, 'X': X,
            'G': G, 'H': H, 'h': h,
            'XL': XL, 'XU': XU,
            'YL': YL, 'YU': YU,
            'Y': []}
    if parallel:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, p, G, H, h, YU, YL, A, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_qcqp, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_qcqp((i, Xi, num_var, Q, p, G, H, h, YU, YL, A, solver))
            Y.append(yt)

    data['Y'] = np.array(Y)
    data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data


def solve_qcqp(args):
    n, Xi, num_var, Q, p, G, H, h, YU, YL, A, solver = args
    y = cp.Variable(num_var)
    constraints = [0.5 * cp.quad_form(y, H[i]) + G[i].T @ y <= h[i] for i in range(H.shape[0])]
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                      constraints + [A @ y == Xi, y <= YU, y >= YL])

    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")

    return y.value


def make_sdp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver):
    np.random.seed(2024)
    """input-output para"""
    XL = input_bound[0]
    XU = input_bound[1]
    YL = output_bound[0]
    YU = output_bound[1]
    """Obj para"""
    Q = np.random.uniform(-1, 1, size=(num_var, num_var))
    Q = (Q + Q.T) / 2
    """Eq para"""
    A = np.random.uniform(-1, 1, size=(num_eq, num_var, num_var))
    A = (A + A.transpose((0, 2, 1))) / 2
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    """Ineq para"""
    y0 = np.random.uniform(-1, 1, size=(num_var, num_var))
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var, num_var))
    G = (G + G.transpose((0, 2, 1))) / 2
    h = np.trace(G @ y0, axis1=1, axis2=2)
    """data set"""
    data = {'Q': Q,
            'A': A, 'X': X,
            'G': G, 'h': h,
            'XL': XL, 'XU': XU,
            'YL': YL, 'YU': YU,
            'Y': []}
    if parallel:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, G, h, YU, YL, A, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_sdp, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_sdp((i, Xi, num_var, Q, G, h, YU, YL, A, solver))
            Y.append(yt)
    A_extend = [np.tril(At) + np.triu(At, 1).T for At in A]
    A_extend = np.array([At[np.tril_indices(num_var)] for At in A_extend])
    G_extend = [np.tril(Gt) + np.triu(Gt, 1).T for Gt in G]
    G_extend = np.array([Gt[np.tril_indices(num_var)] for Gt in G_extend])
    Y_extend = np.array([Yt[np.tril_indices(num_var)] for Yt in Y])
    data['Y'] = np.array(Y)
    data['A'] = np.array(A)
    data['Ye'] = Y_extend
    data['Ae'] = A_extend
    data['Ge'] = G_extend
    data['best_partial'] = find_partial_variable(A_extend, int(num_var * (num_var + 1) / 2), num_eq)
    # A_extend = np.array([A[i].flatten() for i in range(num_eq)])
    # data['best_partial'] = find_partial_variable(A_extend, num_var**2, num_eq)
    return data


def solve_sdp(args):
    n, Xi, num_var, Q, G, h, YU, YL, A, solver = args
    y = cp.Variable((num_var, num_var), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ y)),
                      [cp.trace(A[i] @ y) == Xi[i] for i in range(A.shape[0])] +
                      # [cp.trace(G[i] @ y) <= h[i] for i in range(G.shape[0])] +
                      [y >> 0, y <= YU, y >= YL])
    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0, 0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")

    return y.value


def make_graph_qp(seed, num_node, sparsity, fix_graph, num_var, num_ineq, num_eq, test_size, input_bound, output_bound,
                  parallel, solver):
    if fix_graph:
        graph = [nx.fast_gnp_random_graph(num_node, p=sparsity)]
        edges = [np.array(single_graph.edges()) for single_graph in graph]
    else:
        graph = []
        edges = []
        for _ in range(test_size):
            graph_curr = nx.fast_gnp_random_graph(num_node, p=sparsity)
            edges_curr = np.array(graph_curr.edges())
            graph.append(graph_curr)
            edges.append(edges_curr)

    np.random.seed(seed)
    CL, CU = input_bound
    XL, XU = output_bound
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)

    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    C = np.random.uniform(CL, CU, size=(test_size, num_node, num_eq))
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
    e = np.ones(num_var)

    data = {'Q': Q, 'p': p, 'A': A, 'C': C, 'G': G, 'h': h, 'e': e, 'CL': CL, 'CU': CU, 'XL': XL, 'XU': XU,
            'graph': graph, 'edges': edges, 'xdim': num_var, 'cdim': num_eq, 'ndim': num_node, 'num_ineq': num_ineq,
            'test_size': test_size, }
    if parallel:
        with mp.Pool(processes=n_process) as pool:
            if fix_graph:
                params = [(i, Ci, num_node, edges[0], num_var, Q, p, G, h, XU, XL, A, e, solver) for i, Ci in
                          enumerate(C)]
            else:
                params = [(i, Ci, num_node, edges[i], num_var, Q, p, G, h, XU, XL, A, e, solver) for i, Ci in
                          enumerate(C)]
            X = list(pool.map(solve_graph_qp, params))
    else:
        X = []
        for i, Ci in enumerate(C):
            if fix_graph:
                xt = solve_graph_qp((i, Ci, num_node, edges[0], num_var, Q, p, G, h, XU, XL, A, e, solver))
            else:
                xt = solve_graph_qp((i, Ci, num_node, edges[i], num_var, Q, p, G, h, XU, XL, A, e, solver))
            X.append(xt)
    data['X'] = np.array(X)
    data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data


def solve_graph_qp(args):
    n, Ci, num_node, edges, num_var, Q, p, G, h, XU, XL, A, e, solver = args
    x = cp.Variable((num_node, num_var))

    constraints = []

    for i in range(num_node):
        constraints += [
            G @ x[i] <= h,
            x[i] <= XU,
            x[i] >= XL,
            A @ x[i] == Ci[i]
        ]

    for (i, j) in edges:
        constraints += [
            cp.abs(x[i] - x[j]) <= e
        ]

    obj = 0
    for i in range(num_node):
        obj += cp.quad_form(x[i], Q) + p.T @ x[i]
        obj /= num_node

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(x.value), np.min(x.value), x.value[0, 0:5].T, end='\r')
    except Exception as err:
        print(f"Error solving problem for n={n}: {err}")

    return x.value


def make_jcc_im(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, parallel, solver,
                num_scenario=100):
    np.random.seed(seed)
    XL, XU = input_bound
    YL, YU = output_bound
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    # A = np.zeros([num_eq, num_var])
    # for col in range(num_var):
    #     row = np.random.choice(num_eq)
    #     A[row, col] = 1
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    W = np.random.rand(num_scenario, num_eq) * 0.1
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    # G = np.zeros([num_ineq, num_var])
    # for col in range(num_var):
    #     row = np.random.choice(num_ineq)
    #     G[row, col] = 1
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
    data = {'Q': Q, 'p': p, 'A': A, 'W': W, 'X': X, 'G': G, 'h': h, 'XL': XL, 'XU': XU, 'YL': YL, 'YU': YU, }
    if parallel:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, p, G, h, YU, YL, A, W, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_jcc_im, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_jcc_im((i, Xi, num_var, Q, p, G, h, YU, YL, A, W, solver))
            Y.append(yt)
    data['Y'] = np.array(Y)
    # data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data


def solve_jcc_im(args):
    n, Xi, num_var, Q, p, G, h, YU, YL, A, W, solver = args
    num_scenario = W.shape[0]
    y = cp.Variable(num_var)
    # z = cp.Variable(num_scenario, boolean=True)
    # - (1 - z[i]) * 1e5
    constraints = [y <= YU, y >= YL]
    constraints += [A @ y >= Xi + W[i] for i in range(num_scenario)]
    constraints += [G @ y <= h]
    # constraints.append(cp.sum(z) / num_scenario >= 0.9)
    prob = cp.Problem(cp.Minimize(p.T @ y), constraints)
    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")
    return y.value


def make_power_control(seed, num_node, test_size, parallel, solver):
    np.random.seed(seed)
    graph = nx.fast_gnp_random_graph(num_node, p=1)
    adj = nx.adjacency_matrix(graph).toarray()
    channel_gain = 1 / np.sqrt(2) * np.abs(np.random.rand(test_size, num_node, num_node)
                                           + 1j * np.random.rand(test_size, num_node, num_node))
    weights = np.random.rand(num_node)
    weights = weights / np.sum(weights)
    noises = 0.1
    PL = 0
    PU = 1
    itf = 3
    qos = 1e-5

    data = {'num_node': num_node, 'test_size': test_size, 'adj': adj, 'channel gain': channel_gain, 'weights': weights,
            'noises': noises, 'PL': PL, 'PU': PU, 'itf': itf, 'qos': qos}
    model = init_pc_model(num_node)
    if parallel:
        # with mp.Pool(processes=n_process) as pool:
        #     params = [(model, i, gain, num_node, weights, noises, itf, qos, solver) for i, gain in enumerate(channel_gain)]
        #     P = list(pool.map(solve_power_control, params))
        ...
    else:
        p = []
        for i, gain in enumerate(channel_gain):
            pa = solve_power_control((model, i, gain, num_node, weights, noises, itf, qos, solver))
            p.append(pa)
    data['p'] = np.expand_dims(np.array(p), axis=-1)
    return data


def init_pc_model(num_node):
    model = pyo.AbstractModel()
    model.K = pyo.RangeSet(1, num_node)

    model.gain = pyo.Param(model.K, model.K)
    model.w = pyo.Param(model.K)
    model.n = pyo.Param(model.K)
    model.I = pyo.Param(within=pyo.NonNegativeReals)
    model.QoS = pyo.Param(within=pyo.NonNegativeReals)

    model.p = pyo.Var(model.K, domain=pyo.NonNegativeReals, bounds=(0, 1))

    def obj_expr(m):
        obj = sum(m.w[k] * pyo.log(
            1 + m.gain[k, k] * m.p[k] / (m.n[k] + sum(m.gain[j, k] * m.p[j] for j in m.K if j != k))) / pyo.log(2) for k
                  in m.K)
        return obj

    model.OBJ = pyo.Objective(rule=obj_expr, sense=pyo.maximize)

    def itf_constraint_rule(m, k):
        return sum(m.gain[j, k] * m.p[j] for j in m.K if j != k) <= m.I

    model.c1 = pyo.Constraint(model.K, rule=itf_constraint_rule)

    def QoS_constraint_rule(m, k):
        return pyo.log(
            1 + m.gain[k, k] * m.p[k] / (m.n[k] + sum(m.gain[j, k] * m.p[j] for j in m.K if j != k))) / pyo.log(
            2) >= m.QoS

    model.c2 = pyo.Constraint(model.K, rule=QoS_constraint_rule)

    return model


def solve_power_control(args):
    model, n, channel_gain, num_node, weights, noises, itf, qos, solver = args

    gain_dict = {(i + 1, j + 1): g for i, gain in enumerate(channel_gain) for j, g in enumerate(gain)}
    w_dict = {i + 1: w for i, w in enumerate(weights)}
    n_dict = {i + 1: noises for i in range(num_node)}
    i_dict = {None: itf}
    qos_dict = {None: qos}
    data = {None: {
        'gain': gain_dict,
        'w': w_dict,
        'n': n_dict,
        'I': i_dict,
        'QoS': qos_dict
    }}

    instance = model.create_instance(data)

    # try:
    # if solver == 'ipopt':
    solver = pyo.SolverFactory('ipopt')
    # else:
    #     solver = pyo.SolverFactory(None)
    solver.solve(instance)
    p = [pyo.value(instance.p[i]) for i in instance.K]
    #     print(n, np.max(p), np.min(p), p[0:5], end='\r')
    # except Exception as err:
    #     print(f"Error solving problem for n={n}: {err}")

    return p


from pypower import idx_bus, idx_gen, idx_brch


def make_acopf(seed, num_bus, test_size):
    data = scipy.io.loadmat(
        os.path.dirname(os.getcwd()) + '/ACOPF/data/training_data/ACOPF/{}bus_data.mat'.format(num_bus))
    ppc_mat = scipy.io.loadmat(
        os.path.dirname(os.getcwd()) + '/ACOPF/data/power_grid_cases/{}bus_casefile.mat'.format(num_bus))
    ppc_mat = ppc_mat.get('mpc')
    ppc = {'version': int(ppc_mat['version'][0, 0]), \
           'baseMVA': float(ppc_mat['baseMVA'][0, 0]), \
           'bus': ppc_mat['bus'][0, 0], \
           'gen': ppc_mat['gen'][0, 0], \
           'branch': ppc_mat['branch'][0, 0], \
           'gencost': ppc_mat['gencost'][0, 0]}
    data['ppc'] = ppc
    with open("datasets/acopf/acopf_{}_{}_{}_dataset".format(seed, num_bus, test_size), 'wb') as f:
        pickle.dump(data, f)
    return data


def make_ccacopf(seed, num_bus, test_size):
    data = scipy.io.loadmat(
        os.path.dirname(os.getcwd()) + '/ACOPF/data/ACOPF_L01_variation/acopf_case_{}.mat'.format(num_bus))
    ppc_mat = data.get('mpc')
    ppc = {'version': int(ppc_mat['version'][0, 0]), \
           'baseMVA': float(ppc_mat['baseMVA'][0, 0]), \
           'bus': ppc_mat['bus'][0, 0], \
           'gen': ppc_mat['gen'][0, 0], \
           'branch': ppc_mat['branch'][0, 0], \
           'gencost': ppc_mat['gencost'][0, 0]}
    data['ppc'] = ppc
    np.random.seed(seed)
    sample_index = np.random.choice([i for i in range(data['Dem'].T.shape[0])], test_size, replace=False)
    data['Dem'] = data['Dem'].T[sample_index, :]
    data['Gen'] = data['Gen'].T[sample_index, :]
    data['Vol'] = data['Vol'].T[sample_index, :]
    with open("datasets/acopf/acopf_{}_{}_{}_dataset".format(seed, num_bus, test_size), 'wb') as f:
        pickle.dump(data, f)
    return data


def __main__():
    defaults = {"seed": 1145}
    # solver: mosek, gurobi
    """
    paras: [num_var, n_ineq, n_eq, n_samples]
    """
    # generate_opt(defaults, 'qp', [200, 100, 100, 10000], [-1,1], [-3, 3], paralell=True, solver='mosek')
    # generate_opt(defaults, 'convex_qcqp', [200, 100, 100, 10000], [-1,1], [-3, 3], paralell=True, solver='mosek')
    # generate_opt(defaults, 'socp', [200, 100, 100, 10000], [-1, 1], [-3, 3], paralell=True, solver='mosek')
    # generate_opt(defaults, 'sdp', [400, 0, 20, 10000], [-1, 1], [-3, 3], paralell=True, solver='mosek')
    """
    paras: [num_var, n_ineq, n_eq, n_samples, n_scenario]
    """
    generate_opt(defaults, 'jccim', [400, 100, 100, 10000, 1000], [-1, 1], [-3, 3], parallel=True, solver='mosek')
    """
    paras: [num_var, n_ineq, n_eq, n_samples, num_node, sparsity, fix_topo]
    """
    # generate_opt(defaults, 'graph_qp', [100, 50, 50, 10000, 50, 0.25, 0], [-1, 1], [-3, 3], paralell=True, solver='mosek')
    # generate_opt(defaults, 'graph_qp', [10, 5, 1, 10000, 10, 0.5, 1], [-1, 1], [-3, 3], paralell=False, solver='mosek')
    """
    paras: [num_node, n_samples]
    """
    # generate_opt(defaults, 'power_control', [10, 1024], paralell=False, solver='ipopt')
    """
    paras: [num_bus, n_samples]
    """
    # make_acopf(2023, 30, 10000)
    # make_acopf(2023, 57, 10000)
    # make_acopf(2023, 118, 10000)
    # make_acopf(2023, 200, 10000)
    # make_acopf(2023, 300, 10000)
    # make_acopf(2023, 500, 10000)
    # make_acopf(2023, 793, 10000)
    # make_acopf(2023, 1354, 20000)
    # make_acopf(2023, 2000, 20000)


if __name__ == '__main__':
    __main__()
