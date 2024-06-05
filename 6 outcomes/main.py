import numpy as np
import jax.numpy as jnp
import jax
import parametrize
from distributions import elegant
from tqdm import tqdm

# random parameters using JAX
def random_params(key):
    key, _ = jax.random.split(key)

    # although for the general case we need 216 * 3 parameters, to simplify the load
    # we will take all the states (mmts) to be the same
    return jax.random.uniform(key, (216,)), key

target_p = elegant()
def loss_fn(params):
    return jnp.sqrt(jnp.sum((parametrize.params_to_prob(params) - target_p) ** 2))

# looping code
if __name__ == "__main__":
    close_to_fr = False
    tol = 1e-3 # to see how close Fritz is to parameters
    max_iter = 10000
    current_iter = 0
    key = jax.random.PRNGKey(5)

    # gradient descent parameters
    lr = 1
    epochs = 10000
    # calculate jit version of gradient
    loss_jit = jax.jit(loss_fn)
    grad_ls = jax.jit(jax.grad(loss_fn))

    while True:
        if close_to_fr:
            prob = np.array(parametrize.params_to_prob(params))
            print("Distribution close to target.")
            prob_np = prob[..., 0, 0, 0]
            np.save("final_prob.npy", prob_np)
            break
        elif current_iter > max_iter:
            prob = np.array(parametrize.params_to_prob(params))
            print("Exceeded the maximum number of iterations.")
            prob_np = prob[..., 0, 0, 0]
            np.save("final_prob.npy", prob_np)
            break
        else:
            print("Loop iteration")
            current_iter += 1
            params, key = random_params(key)
            for epoch in tqdm(range(epochs)):
                params = params - lr * grad_ls(params)
                if epoch % 500 == 0:
                    print(f"Epoch {epoch}: loss = {loss_jit(params)}")

            l = loss_jit(params)
            if l < tol:
                close_to_fr = True