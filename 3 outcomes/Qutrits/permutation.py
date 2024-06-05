import numpy as np
import jax.numpy as jnp

def generate_tri_nary_vectors(n):
    vectors = []
    vector = np.zeros(n, dtype=int)

    def generate(vector, i):
        if i == n:
            vectors.append(vector.copy())
            return
        for value in range(3):  # Range changed to 3 for tri-nary
            vector[i] = value
            generate(vector, i + 1)

    generate(vector, 0)
    return np.array(vectors)

# important: when writing the permutation, I am using the convention that p = [p1, p2, ..., pn], this means that
# in the first position of the permuted vector you should put the value in the p1-st position in the original vector,
# NOT that the first value of the original vector goes to the p1-st value of the permuted vector.
def permute_vector(vector, permutation):
    n = vector.shape[0]
    v_p = np.zeros(n)
    for i in range(n):
        v_p[i] = vector[permutation[i] - 1]

    return v_p


def create_matrix(binary_vectors, permutation):
    num_vectors = len(binary_vectors)
    M = np.zeros((num_vectors, num_vectors), dtype=int)

    for i, vector in enumerate(binary_vectors):
        permuted_vector = permute_vector(vector, permutation)
        j = np.where((binary_vectors == permuted_vector).all(axis=1))[0][0]
        M[j][i] = 1

    return jnp.array(M)

if __name__ == "__main__":
    print(create_matrix(generate_tri_nary_vectors(6), np.array([1, 3, 4, 6, 2, 5])).shape)