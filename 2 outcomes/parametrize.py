import numpy as np
import jax.numpy as jnp
import jax
from permutation import *
jax.config.update('jax_enable_x64', True)


# Tamas code for parametrizing Unitaries
def UC(lambda_=None, dtype=jnp.complex_):
    """
    %
    %   COMPOSITE PARAMETERIZATION of U(d) /// version 1.5 /// 12.04.2011
    %
    %   Original Matlab script (retrieved from
    %   https://www.mathworks.com/matlabcentral/fileexchange/30990-composite-parameterization-of-unitary-groups
    %   on 30.09.2022):
    %
    %   Python translation by Tamás Kriváchy and Flavien Hirsch /// 30.09.2022
    %
    %   Usage : UC(lambda)
    %
    %   lambda - dxd real matrix
    %   lambda(a,b) diagonal components a=b - absolute phases for a in [0,2*pi] --> rescaled to [0,1]
    %   lambda(a,b) upper right components a<b - rotations in a-b plane in [0,pi/2] --> rescaled to [0,1]
    %   lambda(a,b) lower left components a>b - relative phases between a-b in [0,2*pi] --> rescaled to [0,1]
    %
    %   References: --- PLEASE CITE THESE PAPERS WHEN USING THIS FILE ---
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'A composite parameterization of unitary groups, density matrices and subspaces'
    %   arXiv:1004.5252 // J. Phys. A: Math. Theor. 43, 385306 (2010); https://doi.org/10.1088/1751-8113/43/38/385306; https://doi.org/10.1088/1751-8113/43/38/385306
    %
    %   Ch.Spengler, M.Huber, B.C.Hiesmayr
    %   'Composite parameterization and Haar measure for all unitary and special unitary groups'
    %   arXiv:1103.3408 // J. Math. Phys. 53, 013501 (2012); https://doi.org/10.1063/1.3672064
    %
    """
    lambda_ = lambda_.astype(dtype)
    # Rescale to [0, 1]
    d = jnp.shape(lambda_)[0]
    i, j = np.indices((d, d))
    lambda_ = jnp.where((i == j) | (i > j), lambda_ * 2 * jnp.pi, lambda_ * jnp.pi / 2) # avoid tracing for JIT
    # for i, j in np.ndindex(lambda_.shape):
    #    if i == j or i > j:
    #       lambda_ = lambda_.at[i, j].set(lambda_[i, j] * 2 * jnp.pi)
            # lambda_[i, j] *= 2 * np.pi
    #    else:
    #        lambda_ = lambda_.at[i, j].set(lambda_[i, j] * jnp.pi / 2)
            # lambda_[i, j] *= np.pi / 2
    unitary = jnp.array([1], dtype=dtype)
    for m in np.arange(d - 1, 0, - 1).reshape(-1):
        i = np.arange(d - m + 1)
        ex1 = jnp.zeros(d - m, dtype=dtype)
        ex2 = jnp.zeros(d - m + 1, dtype=dtype)
        ex2 = jnp.where(i == 0, 1, ex2)
        # ex2[0] = 1
        if len(unitary.shape) == 1:
            unitary = jnp.hstack([ex1, unitary])
        else:
            unitary = jnp.hstack([ex1.reshape(-1, 1), unitary])
        unitary = jnp.vstack([ex2, unitary])
        i, j = np.indices((d - m + 1, d - m + 1))
        for n in np.arange((d - m + 1), 1, - 1).reshape(-1):
            # A = A.astype(dtype=dtype)
            # A = np.eye(d - m + 1, dtype=dtype)
            A = jnp.eye(d - m + 1, dtype=dtype)
            A = jnp.where((i == 0) & (j == 0), jnp.cos(lambda_[m - 1, n + m - 2]), A)
            # A[0, 0] = jnp.cos(lambda_[m - 1, n + m - 2])
            A = jnp.where((i == n - 1) & (j == n - 1), jnp.exp(1j * lambda_[n + m - 2, m - 1]) * jnp.cos(lambda_[m - 1, n + m - 2]), A)
            # A[n - 1, n - 1] = np.exp(1j * lambda_[n + m - 2, m - 1]) * np.cos(lambda_[m - 1, n + m - 2])
            A = jnp.where((i == n - 1) & (j == 0), - jnp.exp(1j * lambda_[n + m - 2, m - 1]) * jnp.sin(lambda_[m - 1, n + m - 2]), A)
            # A[n - 1, 0] = - np.exp(1j * lambda_[n + m - 2, m - 1]) * np.sin(lambda_[m - 1, n + m - 2])
            A = jnp.where((i == 0) & (j == n - 1), jnp.sin(lambda_[m - 1, n + m - 2]), A)
            # A[0, n - 1] = np.sin(lambda_[m - 1, n + m - 2])
            unitary = A @ unitary
    # for k in np.arange(0, d).reshape(-1):
    #    unitary[:, k] = unitary[:, k] * np.exp(1j * lambda_[k, k])
    k = jnp.arange(0, d)
    return unitary * jnp.exp(1j * lambda_[k, k][:, None])


# parametrizing states: function that gets as input 16 numbers, and generates the matrix of the state
def state_pmt(params):
    U = UC(params.reshape((4, 4)))
    vector = jnp.zeros(4)
    i = np.arange(4)
    vector = jnp.where(i == 0, 1, vector) # the vector we are going to rotate is |0> (computational basis)

    return jnp.array(jnp.dot(jnp.transpose(jnp.conj(U)),
                  jnp.dot(jnp.outer(vector, vector), U))) # skipping conjugation because vector has real coefficients


# parametrizing projectors
def array_projectors(U):
    # Initialize the array to store the resulting matrices
    result_matrices = jnp.zeros((4, U.shape[0], U.shape[1]), dtype=U.dtype)
    i, j, k = np.indices((4, U.shape[0], U.shape[1]))

    # Iterate over the four canonical vectors
    l, m = np.indices((U.shape[0], 1))
    for t in range(4):
        # Create the t-th canonical vector
        canonical_vector = jnp.zeros((U.shape[0], 1), dtype=U.dtype)
        canonical_vector = jnp.where(l == t, 1, canonical_vector)

        # Calculate the matrix product U^dagger * (outer product of the t-th canonical vector with itself) * U
        result_matrices = jnp.where(i == t, jnp.dot(jnp.transpose(jnp.conj(U)), jnp.dot(jnp.outer(canonical_vector, jnp.conj(canonical_vector)), U)), result_matrices)
        # result_matrices[t] = jnp.dot(jnp.transpose(jnp.conj(U)), jnp.dot(jnp.outer(canonical_vector, jnp.conj(canonical_vector)), U))

    return jnp.array(result_matrices)


# to parametrize projectors we have to pass 20 parameters: 16 for U and 4 for the phase transition probabilities p(a|t)
def proj_pmt(params):
    matrix_params = params[:16]
    U_t = array_projectors(UC(matrix_params.reshape((4, 4))))

    Pi_1 = jnp.zeros((4, 4), dtype=jnp.complex_)
    Pi_2 = jnp.zeros((4, 4), dtype=jnp.complex_)
    trans_prob = jnp.reshape(jnp.abs(params[-8:]), (2,4))
    col_sums = jnp.sum(trans_prob, axis=0)
    trans_upd = jnp.array(trans_prob / col_sums, dtype=jnp.complex_)

    for j in range(4):
        Pi_1 = trans_upd[0][j] * U_t[j] + Pi_1
        Pi_2 = trans_upd[1][j] * U_t[j] + Pi_2

#    for j in range(4):
#        if i == 0:
#            Pi[i] += trans_upd[j] * U_t[j]
#        else:
#            Pi[i] += (1 - trans_upd[j]) * U_t[j]

    return jnp.stack([Pi_1, Pi_2])


permutation = np.array([1, 3, 4, 6, 2, 5])  # to go from the space that Pi acts on to the one Rho acts on
# Pi: Ab, Ac, Ba, Bc, Ca, Cb -> Rho: Ab, Ba, Bc, Cb, Ac, Ca
M = create_matrix(generate_binary_vectors(6), permutation)


# function that, given 108 parameters, return the quantum probability distribution p(a,b,c)
def params_to_prob(params):
    rho_ab = state_pmt(params[:16])
    rho_bc = state_pmt(params[16:32])
    rho_ac = state_pmt(params[32:48])
    Pi_A = proj_pmt(params[48:72])
    Pi_B = proj_pmt(params[72:96])
    Pi_C = proj_pmt(params[96:])

    P = jnp.zeros((2, 2, 2))
    i, j, k = np.indices((2,2,2))
    for a, b, c in np.ndindex(*P.shape):
        res = jnp.kron(jnp.kron(Pi_A[a], Pi_B[b]), Pi_C[c])
        # TODO: implement einsum
        res = M @ res
        res = jnp.kron(rho_ab, jnp.kron(rho_bc, rho_ac)) @ res
        res = np.transpose(M) @ res

        tr = jnp.trace(res)
        P = jnp.where((i == a) & (j == b) & (k == c), jnp.real(tr), P)
        # P[a, b, c] = jnp.real(tr)

    p = jnp.zeros((2, 2, 2, 1, 1, 1))
    p = jnp.broadcast_to(P[..., None, None, None], P.shape + (1, 1, 1))

    return jnp.array(p)