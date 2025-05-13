import torch
import numpy as np
from scipy.linalg import eigh, inv  # For eigenvalues of symmetric matrices, inverse
from scipy.optimize import brentq  # For root finding
import warnings


# Helper for Frobenius inner product
def frobenius_inner_product(M1, M2):
    """Computes the Frobenius inner product of two matrices M1 and M2."""
    return np.sum(M1 * M2)


# Helper for Frobenius norm
def frobenius_norm(M):
    """Computes the Frobenius norm of a matrix M."""
    return np.linalg.norm(M, 'fro')


def is_positive_definite(M, tol=1e-9):
    """
    Checks if a matrix M is symmetric and positive definite.
    A matrix is positive definite if it is symmetric and all its eigenvalues are positive.
    """
    if not isinstance(M, np.ndarray) or M.ndim != 2 or M.shape[0] != M.shape[1]:
        warnings.warn("is_positive_definite received non-square matrix input.", RuntimeWarning)
        return False
    # Check for symmetry
    if not np.allclose(M, M.T):
        warnings.warn("Matrix is not symmetric, cannot be positive definite.", RuntimeWarning)
        return False
    try:
        # eigh is for Hermitian (symmetric if real) matrices
        eigenvalues = eigh(M, eigvals_only=True)
        return np.all(eigenvalues > tol)
    except np.linalg.LinAlgError:
        # LinAlgError can occur for various reasons, e.g. if matrix is not numerically Hermitian
        return False


# Target log probability (up to a constant)
def log_target_density(X, A, B, C, pd_check_tol=1e-9):
    """
    Calculates the log of the target probability density (up to a constant).
    log P(X) proportional to: -<A, X> - ||X - B||_F - ||X - C||_F^2 + log(det(X))
    """
    if not is_positive_definite(X, tol=pd_check_tol):
        return -np.inf

    # np.linalg.slogdet returns (sign, log(abs(det(X))))
    # For a positive definite matrix, sign must be 1.
    sign, log_det_X_val = np.linalg.slogdet(X)

    if sign <= 0 or np.isinf(log_det_X_val) or np.isnan(log_det_X_val):
        return -np.inf

    term_A_X = frobenius_inner_product(A, X)
    term_X_B_norm = frobenius_norm(X - B)
    term_X_C_norm_sq = frobenius_norm(X - C) ** 2

    return -term_A_X - term_X_B_norm - term_X_C_norm_sq + log_det_X_val


# Function to find lambda bounds for positive definiteness
def get_pd_lambda_bounds(X_current, D_matrix, tol=1e-9,
                         search_scale_pos=1.0, search_scale_neg=1.0,
                         max_expand_iters=30, bisection_tol=1e-7):
    """
    Finds interval (lambda_min_pd, lambda_max_pd) where X_current + lambda * D_matrix
    has its minimum eigenvalue > tol.
    Assumes X_current and D_matrix are symmetric.
    """

    def min_eigenvalue_objective(lmbda):
        """Returns min_eigenvalue(X_current + lmbda * D_matrix) - tol."""
        M = X_current + lmbda * D_matrix
        try:
            # subset_by_index=[0,0] gets the smallest eigenvalue for eigh
            return eigh(M, eigvals_only=True, subset_by_index=[0, 0])[0] - tol
        except np.linalg.LinAlgError:
            # If matrix becomes non-Hermitian or other errors during eigenvalue computation
            return -np.inf  # Indicates failure, effectively not PD by margin tol

    # Check if X_current itself satisfies the condition
    obj_at_zero = min_eigenvalue_objective(0)
    if obj_at_zero <= 0:
        # This means X_current's min_eigenvalue is already <= tol. No room to move.
        warnings.warn(f"X_current (min_eig={eigh(X_current, eigvals_only=True, subset_by_index=[0, 0])[0]}) "
                      f"is not positive definite by the required margin {tol}. "
                      f"Cannot find PD lambda bounds.", RuntimeWarning)
        return 0.0, 0.0

        # Find lambda_max_pd (positive lambda where min_eigenvalue - tol crosses zero)
    lambda_max_pd = np.inf
    # Check if moving in positive lambda direction decreases the min eigenvalue
    if min_eigenvalue_objective(search_scale_pos * 1e-4) < obj_at_zero:
        low_bracket = 0.0
        high_bracket = search_scale_pos
        found_upper_crossing_bracket = False
        for _ in range(max_expand_iters):  # Expand search interval for high_bracket
            if min_eigenvalue_objective(high_bracket) <= 0:
                found_upper_crossing_bracket = True
                break
            high_bracket *= 1.5

        if found_upper_crossing_bracket:
            try:  # Find root using bisection
                lambda_max_pd = brentq(min_eigenvalue_objective, low_bracket, high_bracket,
                                       xtol=bisection_tol, rtol=bisection_tol)
            except ValueError:  # brentq fails if f(a) and f(b) don't have opposite signs
                warnings.warn(f"Brentq failed for lambda_max_pd. f(low={min_eigenvalue_objective(low_bracket)}), "
                              f"f(high={min_eigenvalue_objective(high_bracket)}). Using high_bracket if non-PD.",
                              RuntimeWarning)
                if min_eigenvalue_objective(high_bracket) <= 0: lambda_max_pd = high_bracket
                # else lambda_max_pd remains np.inf (PD in this range)
        # else: min_eigenvalue objective remains positive up to high_bracket, lambda_max_pd is np.inf

    # Find lambda_min_pd (negative lambda where min_eigenvalue - tol crosses zero)
    lambda_min_pd = -np.inf
    if min_eigenvalue_objective(-search_scale_neg * 1e-4) < obj_at_zero:
        high_bracket = 0.0  # Lambda = 0
        low_bracket = -search_scale_neg  # Negative lambda
        found_lower_crossing_bracket = False
        for _ in range(max_expand_iters):  # Expand search interval for low_bracket (more negative)
            if min_eigenvalue_objective(low_bracket) <= 0:
                found_lower_crossing_bracket = True
                break
            low_bracket *= 1.5

        if found_lower_crossing_bracket:
            try:  # Find root
                lambda_min_pd = brentq(min_eigenvalue_objective, low_bracket, high_bracket,
                                       xtol=bisection_tol, rtol=bisection_tol)
            except ValueError:
                warnings.warn(f"Brentq failed for lambda_min_pd. f(low={min_eigenvalue_objective(low_bracket)}), "
                              f"f(high={min_eigenvalue_objective(high_bracket)}). Using low_bracket if non-PD.",
                              RuntimeWarning)
                if min_eigenvalue_objective(low_bracket) <= 0: lambda_min_pd = low_bracket
                # else lambda_min_pd remains -np.inf

    return lambda_min_pd, lambda_max_pd


# Grid-based sampling for lambda
def sample_lambda_grid(lambda_lower, lambda_upper, X_current, D_matrix, A, B, C,
                       pd_check_tol, num_points=200):
    """Samples lambda from a 1D slice of the target distribution using a grid approximation."""
    if lambda_lower >= lambda_upper - 1e-9:  # Interval too small or invalid
        return None

    lambdas = np.linspace(lambda_lower, lambda_upper, num_points)

    log_probs = np.array([log_target_density(X_current + l * D_matrix, A, B, C, pd_check_tol)
                          for l in lambdas])

    # Filter out -np.inf or NaN log_probs
    valid_indices = ~np.isinf(log_probs) & ~np.isnan(log_probs)
    if not np.any(valid_indices):
        warnings.warn("No valid (non-infinite, non-NaN) log-probabilities found for lambda sampling.", RuntimeWarning)
        return None

    lambdas = lambdas[valid_indices]
    log_probs = log_probs[valid_indices]

    if len(lambdas) == 0:  # Should be caught by previous check
        return None

    # Convert log_probs to probs, carefully to avoid underflow/overflow
    max_log_prob = np.max(log_probs)
    probs = np.exp(log_probs - max_log_prob)
    sum_probs = np.sum(probs)

    if sum_probs < 1e-100 or np.isnan(sum_probs):  # All probs numerically zero or NaN
        warnings.warn("All probabilities for lambda sampling are effectively zero or NaN. "
                      "Choosing lambda uniformly from the valid range.", RuntimeWarning)
        return np.random.choice(lambdas) if len(lambdas) > 0 else None

    probs /= sum_probs  # Normalize

    return np.random.choice(lambdas, p=probs)


# Main Hit-and-Run Sampler
def hit_and_run_matrix_sampler(X_init, A, B, C, D_constraints, c_constraints,
                               n_samples, burn_in=100, thin=1,
                               pd_boundary_tol=1e-8,
                               pd_acceptance_tol=1e-7,
                               lambda_search_scale=1.0,
                               random_seed=None):
    """
    Generates samples from the specified matrix distribution using Hit-and-Run.

    Args:
        X_init (np.ndarray): Initial symmetric, positive definite matrix satisfying constraints.
        A, B, C (np.ndarray): Symmetric matrices defining the target distribution.
        D_constraints (list of np.ndarray): List of symmetric D_i matrices for linear constraints.
        c_constraints (list of float): List of c_i values for linear constraints.
        n_samples (int): Number of samples to generate (after burn-in and thinning).
        burn_in (int): Number of initial samples to discard.
        thin (int): Thinning factor.
        pd_boundary_tol (float): Tolerance for min_eigenvalue in get_pd_lambda_bounds.
                                 Defines how close to singular the boundary search goes.
        pd_acceptance_tol (float): Tolerance for checking positive definiteness of accepted samples.
                                   Should be >= pd_boundary_tol.
        lambda_search_scale (float): Initial scale for searching lambda bounds for PD.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        list of np.ndarray: Generated samples.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Validate X_init
    if not isinstance(X_init, np.ndarray) or X_init.ndim != 2 or X_init.shape[0] != X_init.shape[1]:
        raise ValueError("X_init must be a square matrix.")
    if not np.allclose(X_init, X_init.T):  # Check symmetry
        raise ValueError("X_init must be a symmetric matrix.")
    if not is_positive_definite(X_init, tol=pd_acceptance_tol):
        min_eig_X_init = eigh(X_init, eigvals_only=True, subset_by_index=[0, 0])[0] if np.allclose(X_init,
                                                                                                   X_init.T) else "N/A (not symm)"
        raise ValueError(f"X_init (min_eig={min_eig_X_init}) must be positive definite "
                         f"with tolerance {pd_acceptance_tol}.")
    for i, (D_i, c_i) in enumerate(zip(D_constraints, c_constraints)):
        if not np.allclose(D_i, D_i.T):
            raise ValueError(f"D_constraint matrix {i} must be symmetric.")
        if frobenius_inner_product(D_i, X_init) < c_i - pd_acceptance_tol:
            raise ValueError(f"X_init does not satisfy linear constraint {i}: "
                             f"<D_{i}, X> ({frobenius_inner_product(D_i, X_init)}) < c_{i} ({c_i}).")

    dim = X_init.shape[0]
    X_current = X_init.copy()
    samples = []

    total_iterations = burn_in + n_samples * thin
    accepted_moves_post_burn_in = 0

    for iteration in range(total_iterations):
        # 1. Choose a random symmetric direction matrix D_matrix
        D_rand_raw = np.random.randn(dim, dim)
        D_matrix = (D_rand_raw + D_rand_raw.T) / 2
        norm_D = frobenius_norm(D_matrix)

        if norm_D < 1e-12:  # Avoid division by zero or effectively zero direction
            D_matrix = np.zeros_like(D_matrix)
        else:
            D_matrix /= norm_D

        if np.allclose(D_matrix, 0):  # If direction is zero, no move possible
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                samples.append(X_current.copy())  # Store current point if no move
            if len(samples) >= n_samples: break
            continue

        # 2. Find lambda bounds for positive definiteness (min_eig > pd_boundary_tol)
        lambda_pd_min, lambda_pd_max = get_pd_lambda_bounds(
            X_current, D_matrix, tol=pd_boundary_tol,
            search_scale_pos=lambda_search_scale, search_scale_neg=lambda_search_scale
        )

        # 3. Find lambda bounds from linear constraints <D_i, X(lambda)> >= c_i
        lambda_linear_min, lambda_linear_max = -np.inf, np.inf
        no_valid_linear_interval = False
        for D_i, c_i in zip(D_constraints, c_constraints):
            val_Di_Dmatrix = frobenius_inner_product(D_i, D_matrix)
            val_Di_Xcurrent = frobenius_inner_product(D_i, X_current)

            # Effective c_i for boundary check, slightly inset for robustness
            # c_i_eff = c_i - pd_boundary_tol # Ensure X(lambda) is strictly > c_i
            # Using c_i directly is standard, relies on lambda sampling not hitting exact boundary
            c_i_eff = c_i

            if np.abs(val_Di_Dmatrix) < 1e-9:  # Direction D_matrix is nearly parallel to constraint plane
                if val_Di_Xcurrent < c_i_eff - pd_boundary_tol:  # Current point violates or line is infeasible
                    no_valid_linear_interval = True;
                    break
            elif val_Di_Dmatrix > 0:  # <D_i, D_matrix> is positive
                lambda_linear_min = max(lambda_linear_min, (c_i_eff - val_Di_Xcurrent) / val_Di_Dmatrix)
            else:  # <D_i, D_matrix> is negative
                lambda_linear_max = min(lambda_linear_max, (c_i_eff - val_Di_Xcurrent) / val_Di_Dmatrix)

        if no_valid_linear_interval or lambda_linear_min > lambda_linear_max + pd_boundary_tol:
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                samples.append(X_current.copy())
            if len(samples) >= n_samples: break
            continue

        # 4. Combine bounds to get final feasible interval for lambda
        final_lambda_min = max(lambda_pd_min, lambda_linear_min)
        final_lambda_max = min(lambda_pd_max, lambda_linear_max)

        if final_lambda_min >= final_lambda_max - 1e-9:  # Interval is empty or too small
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                samples.append(X_current.copy())
            if len(samples) >= n_samples: break
            continue

        # 5. Sample lambda from 1D distribution on [final_lambda_min, final_lambda_max]
        lambda_chosen = sample_lambda_grid(final_lambda_min, final_lambda_max,
                                           X_current, D_matrix, A, B, C,
                                           pd_check_tol=pd_boundary_tol)

        X_next_candidate = X_current  # Default to staying put if sampling fails
        made_move_this_step = False
        if lambda_chosen is not None:
            proposed_X = X_current + lambda_chosen * D_matrix
            # Final check for the proposed point using stricter acceptance tolerance
            if is_positive_definite(proposed_X, tol=pd_acceptance_tol):
                all_linear_constraints_met = True
                for D_i, c_i in zip(D_constraints, c_constraints):
                    if frobenius_inner_product(D_i, proposed_X) < c_i - pd_acceptance_tol:
                        all_linear_constraints_met = False;
                        break
                if all_linear_constraints_met:
                    X_next_candidate = proposed_X
                    made_move_this_step = True
                    if iteration >= burn_in:
                        accepted_moves_post_burn_in += 1
            # else: proposed_X is not valid, X_next_candidate remains X_current

        X_current = X_next_candidate

        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            samples.append(X_current.copy())

        if len(samples) >= n_samples:
            break

        # Progress indicator
        if iteration > 0 and total_iterations > 10 and iteration % (total_iterations // 10) == 0:
            current_acceptance_rate = 0
            if iteration >= burn_in and (iteration - burn_in + 1) > 0:
                current_acceptance_rate = accepted_moves_post_burn_in / (iteration - burn_in + 1)
            print(f"Iteration {iteration}/{total_iterations}. "
                  f"Acceptance rate (post-burn-in): {current_acceptance_rate:.2f}")

    final_acceptance_rate = 0
    if (n_samples * thin) > 0:
        final_acceptance_rate = accepted_moves_post_burn_in / (n_samples * thin)
    print(f"Sampling finished. Final acceptance rate (post-burn-in): {final_acceptance_rate:.2f}")
    return samples


if __name__ == '__main__':
    # Example usage: (User needs to define A, B, C, D_constraints, c_constraints, and a valid X_init)
    dim = 2  # Example dimension

    # Define symmetric matrices for the distribution
    A_mat = np.array([[1, 0.5], [0.5, 1]])
    B_mat = np.eye(dim) * 2
    C_mat = np.eye(dim) * -1

    # Define linear constraints: <D_i, X> >= c_i
    # Example: Constraint X[0,0] >= 0.1
    D1 = np.zeros((dim, dim));
    D1[0, 0] = 1.0
    c1 = 0.1
    # Example: Constraint tr(X) >= 1.0 (sum of diagonal elements)
    D2 = np.eye(dim)
    c2 = 1.0

    D_constraints_list = [D1, D2]
    c_constraints_list = [c1, c2]

    # Define a valid initial X_init: symmetric, positive definite, and satisfies constraints
    # This is crucial and often problem-specific.
    # For this example, let's try X_init = Identity, then check/adjust.
    X_initial = np.array([[0.5, 0.1], [0.1, 0.6]])  # np.eye(dim)

    # Ensure X_initial meets criteria (this is a basic check, more robust feasibility needed for general cases)
    if not is_positive_definite(X_initial, tol=1e-7):  # Using a typical acceptance tolerance
        print(f"Error: Example X_initial is not PD. Min eigenvalue: {eigh(X_initial, eigvals_only=True)[0]}")
        exit()
    for i_constr, (D_i, c_i) in enumerate(zip(D_constraints_list, c_constraints_list)):
        if frobenius_inner_product(D_i, X_initial) < c_i - 1e-7:
            print(f"Error: Example X_initial fails linear constraint {i_constr}. "
                  f"Value: {frobenius_inner_product(D_i, X_initial)}, Required: >={c_i}")
            exit()
    print(
        f"Using initial X_init (min_eig={eigh(X_initial, eigvals_only=True, subset_by_index=[0, 0])[0]}):\n{X_initial}")

    # Suppress some runtime warnings that might occur during boundary searches if desired, after debugging.
    # warnings.simplefilter("ignore", RuntimeWarning)

    try:
        generated_samples = hit_and_run_matrix_sampler(
            X_init=X_initial, A=A_mat, B=B_mat, C=C_mat,
            D_constraints=D_constraints_list, c_constraints=c_constraints_list,
            n_samples=50,  # Number of samples to collect
            burn_in=20,  # Number of burn-in samples
            thin=1,  # Thinning factor
            pd_boundary_tol=1e-9,  # Tolerance for eigenvalue > tol in get_pd_lambda_bounds
            pd_acceptance_tol=1e-8,  # Stricter tolerance for accepting a point as PD
            lambda_search_scale=1.0,  # Initial search scale for lambda
            random_seed=123
        )
        print(f"\nGenerated {len(generated_samples)} samples.")
        if generated_samples:
            print("First sample's min eigenvalue:", eigh(generated_samples[0], eigvals_only=True)[0])
            print("Last sample's min eigenvalue:", eigh(generated_samples[-1], eigvals_only=True)[0])

            # Basic validation of generated samples
            invalid_sample_count = 0
            for k, sample_X in enumerate(generated_samples):
                if not is_positive_definite(sample_X, tol=1e-8):  # pd_acceptance_tol
                    invalid_sample_count += 1
                    print(f"Warning: Sample {k} is not PD! Min eig: {eigh(sample_X, eigvals_only=True)[0]}")
                for D_idx, (D_i_val, c_i_val) in enumerate(zip(D_constraints_list, c_constraints_list)):
                    if frobenius_inner_product(D_i_val, sample_X) < c_i_val - 1e-8:  # pd_acceptance_tol
                        invalid_sample_count += 1
                        print(f"Warning: Sample {k} violates linear constraint {D_idx}! "
                              f"Value={frobenius_inner_product(D_i_val, sample_X)}, Required >={c_i_val}")
            if invalid_sample_count == 0:
                print("All generated samples passed basic validity checks.")
            else:
                print(f"Found {invalid_sample_count} issues in {len(generated_samples)} samples.")

    except ValueError as e:
        print(f"ValueError during sampling: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
