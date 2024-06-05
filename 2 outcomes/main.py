import numpy as np
import jax.numpy as jnp
import jax
from inflation import InflationProblem, InflationLP
from cvxpy_feas import CVXPY_feasibility_as_optimisation
import parametrize
import inequalities
from tqdm import tqdm

# random parameters using JAX
def random_params(key):
    key, _ = jax.random.split(key)

    return jax.random.uniform(key, (4, 120)), key


# params to value functions
def params_to_val1(params):
    return inequalities.ineq_1(parametrize.params_to_prob(params))

def params_to_val2(params):
    return inequalities.ineq_2(parametrize.params_to_prob(params))

def params_to_val3(params):
    return inequalities.ineq_3(parametrize.params_to_prob(params))

def params_to_val4(params):
    return inequalities.ineq_4(parametrize.params_to_prob(params))

triangle = InflationProblem(dag={"rho_AB": ["A", "B"],
                                     "rho_BC": ["B", "C"],
                                     "rho_AC": ["A", "C"]},
                                outcomes_per_party=(2, 2, 2),
                                settings_per_party=(1, 1, 1),
                                inflation_level_per_source=(2, 2, 2),
                                verbose=1)
lp = InflationLP(triangle, verbose=1)
# lp_as_cvxpy_instance = CVXPY_feasibility_as_optimisation(lp)
# function that determines whether a given p is classical using inflation
def is_classical(p):
    # lp_as_cvxpy_instance.set_p(p)
    lp.set_distribution(p)
    lp.verbose = 1
    lp.solve()

    return lp.success
    # max_lambda, cert_symbolic, cert_dictionary = lp_as_cvxpy_instance.solve(verbose=False)

    #if max_lambda > 0:
    #    return True
    #else:
    #    return False


# looping code
if __name__ == "__main__":
    quantum_non_cl = False
    max_iter = 10000  # maximum number of iterations
    current_iter = 0
    non_cl_iter = 0  # number of iterations where the random distribution was non-classical
    key = jax.random.PRNGKey(3)
    j = 0 # to store which params to save later

    # gradient descent parameters
    lr = jnp.array([1e-3, 1e-6, 1e-3, 1e-9])
    epochs = 15000
    # calculate jit version of gradient
    grad_1 = jax.jit(jax.grad(params_to_val1))
    grad_2 = jax.jit(jax.grad(params_to_val2))
    grad_3 = jax.jit(jax.grad(params_to_val3))
    grad_4 = jax.jit(jax.grad(params_to_val4))

    while True:
        if quantum_non_cl:
            print("Violated some inequality")
            jnp.save("final_params.npy", params[j])
            print(f"Is this problem feasible? {is_classical(np.array(parametrize.params_to_prob(params[j])))}")
            # print("Found a quantum non-classical point")
            break
        elif current_iter > max_iter:
            print("Exceeded the maximum number of iterations.")
            break
        else:
            print("Loop iteration")
            current_iter += 1
            params, key = random_params(key)
            for epoch in tqdm(range(epochs)):
                params = params.at[0].set(params[0] - lr[0] * grad_1(params[0]))
                params = params.at[1].set(params[1] - lr[1] * grad_2(params[1]))
                params = params.at[2].set(params[2] - lr[2] * grad_3(params[2]))
                params = params.at[3].set(params[3] - lr[3] * grad_4(params[3]))

                if epoch % 500 == 0:
                    print(f"Epoch {epoch}: ineq_1 = {inequalities.ineq_1(parametrize.params_to_prob(params[0]))}")
                    print(f"Epoch {epoch}: ineq_2 = {inequalities.ineq_1(parametrize.params_to_prob(params[1]))}")
                    print(f"Epoch {epoch}: ineq_3 = {inequalities.ineq_1(parametrize.params_to_prob(params[2]))}")
                    print(f"Epoch {epoch}: ineq_4 = {inequalities.ineq_1(parametrize.params_to_prob(params[3]))}")

            for i in range(4):
                in1 = inequalities.ineq_1(parametrize.params_to_prob(params[0]))
                in2 = inequalities.ineq_2(parametrize.params_to_prob(params[1]))
                in3 = inequalities.ineq_3(parametrize.params_to_prob(params[2]))
                in4 = inequalities.ineq_4(parametrize.params_to_prob(params[3]))
                # if not is_classical(np.array(parametrize.params_to_prob(params[i]))):
                #    quantum_non_cl = True
                #    j = i
            if in1 < 0:
                quantum_non_cl = True
                j = 0
            elif in2 < 0:
                quantum_non_cl = True
                j = 1
            elif in3 < 0:
                quantum_non_cl = True
                j = 2
            elif in4 < 0:
                quantum_non_cl = True
                j = 3
