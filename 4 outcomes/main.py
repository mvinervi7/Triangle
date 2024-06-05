import numpy as np
import jax.numpy as jnp
import jax
from inflation import InflationProblem, InflationLP
from cvxpy_feas import CVXPY_feasibility_as_optimisation
import parametrize
from distributions import P_Fritz, elegant
from tqdm import tqdm

# random parameters using JAX
def random_params(key):
    key, _ = jax.random.split(key)

    return jax.random.uniform(key, (144,)), key # we now need 144 parameters instead of 108

triangle = InflationProblem(dag={"rho_AB": ["A", "B"],
                                     "rho_BC": ["B", "C"],
                                     "rho_AC": ["A", "C"]},
                                outcomes_per_party=(4, 4, 4),
                                settings_per_party=(1, 1, 1),
                                inflation_level_per_source=(2, 2, 2),
                                verbose=1)
lp = InflationLP(triangle, verbose=1)
lp_as_cvxpy_instance = CVXPY_feasibility_as_optimisation(lp)

# function that determines whether a given p is classical using inflation
def is_classical(p):
    lp_as_cvxpy_instance.set_p(p)
    max_lambda, cert_symbolic, cert_dictionary = lp_as_cvxpy_instance.solve(verbose=False)

    if max_lambda > 0:
        return True
    else:
        return False

target_p = elegant()
def loss_fn(params):
    return jnp.sum((parametrize.params_to_prob(params) - target_p) ** 2)

# looping code
if __name__ == "__main__":
    quantum_non_cl = False
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
            print("Distribution close to Fritz")
            print(f"Is the distribution classical? {is_classical(prob)}")
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