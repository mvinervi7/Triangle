import qutip as qt
import numpy as np
import jax.numpy as jnp

def elegant():
    prob = np.zeros((6, 6, 6, 1, 1, 1))
    for a, b, c in np.ndindex(6, 6, 6):
        if a == b == c:
            prob[a, b, c, 0, 0, 0] = 0.19/6
        elif (a == b and a != c) or (b == c and b != a) or (a == c and a != b):
            prob[a, b, c, 0, 0, 0] = 0.15/90
        elif a != b and a != c and b != c:
            prob[a, b, c, 0, 0, 0] = 0.66/120

    return jnp.array(prob)


if __name__ == '__main__':
    print(jnp.sum(elegant()))