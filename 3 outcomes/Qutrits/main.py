import jax
import jax.numpy as jnp
import numpy as np
import parametrize

def random_params(key):
    key, _ = jax.random.split(key)

    return jax.random.uniform(key, (567,)), key

def loss_fn(params, target_p):
    return jnp.sqrt(jnp.sum((parametrize.params_to_prob(params) - target_p) ** 2))

def elegant(s111, s112, s123):
    prob = np.zeros((3, 3, 3, 1, 1, 1))
    for a, b, c in np.ndindex(3, 3, 3):
        if a == b == c:
            prob[a, b, c, 0, 0, 0] = s111 / 3.0
        elif (a == b and a != c) or (b == c and b != a) or (a == c and a != b):
            prob[a, b, c, 0, 0, 0] = s112 / 18.0
        elif a != b and a != c and b != c:
            prob[a, b, c, 0, 0, 0] = s123 / 6.0

    return jnp.array(prob)

if __name__ == '__main__':
    key = jax.random.PRNGKey(5)
    lr_initial = 1
    lr_final = 1e-1
    epochs = 10000
    loss_jit = jax.jit(loss_fn)
    grad_ls = jax.jit(jax.grad(loss_fn))

    # 0.4, 0.32, 0.28
    target = elegant(0.4, 0.32, 0.28)

    # 0.4, 0.36, 0.24
