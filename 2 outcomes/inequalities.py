import numpy as np
import jax.numpy as jnp
import jax

# note: I initially had a concise function to which I passed as arguments the type of expected value I was going to pass
# (e.g. if I wanted <AC> I could pass [1,0,1], but I was unable to then jit my function, so had to resort to defining
# myriad very similar functions

# also tried to implement jax.lax.scan to avoid for loops, which really mess up jax's efficiency. in this code it won't be so
# relevant since we have not so many elements, but can be interesting (1) for compatibility reasons with jit and
# (2) other bigger loops

def generate_indices(shape):
    # Generate ranges for each dimension
    ranges = [jnp.arange(dim) for dim in shape]
    # Create mesh grid using broadcasting
    mesh = jnp.meshgrid(*ranges, indexing='ij')
    # Stack the mesh grid and reshape to get the desired output
    indices = jnp.stack(mesh, axis=-1).reshape(-1, len(shape))
    return jnp.array(indices)

def exp_val_A(p):
    # function that takes in (current state, next element in the array over which it's iterating) ->
    # (updated state, observed state). it will then stack all the observed states, but we don't need that
    def scan_fn(acc_prob, indices):
        acc_prob += jax.lax.cond(indices[0] == 1, lambda _: p[tuple(indices)], lambda _: 0.0, None)
        return acc_prob, None

    prob_1 = 0
    prob_1, _ = jax.lax.scan(scan_fn, prob_1, generate_indices(p.shape))
    return -1.0 * (1 - prob_1) + 1.0 * prob_1

def exp_val_B(p):
    def scan_fn(acc_prob, indices):
        acc_prob += jax.lax.cond(indices[1] == 1, lambda _: p[tuple(indices)], lambda _: 0.0, None)
        return acc_prob, None

    prob_1 = 0
    prob_1, _ = jax.lax.scan(scan_fn, prob_1, generate_indices(p.shape))
    return -1.0 * (1 - prob_1) + 1.0 * prob_1


def exp_val_C(p):
    def scan_fn(acc_prob, indices):
        acc_prob += jax.lax.cond(indices[2] == 1, lambda _: p[tuple(indices)], lambda _: 0.0, None)
        return acc_prob, None

    prob_1 = 0
    prob_1, _ = jax.lax.scan(scan_fn, prob_1, generate_indices(p.shape))
    return -1.0 * (1 - prob_1) + 1.0 * prob_1


def exp_val_AB(p):
    def scan_fn(acc_prob, indices):
        condition = (indices[0] == indices[1])
        acc_prob += jax.lax.cond(condition, lambda _: p[tuple(indices)], lambda _: 0.0, None)
        return acc_prob, None

    prob_1 = 0
    prob_1, _ = jax.lax.scan(scan_fn, prob_1, generate_indices(p.shape))

    return -1.0 * (1 - prob_1) + 1.0 * prob_1
def exp_val_AC(p):
    def scan_fn(acc_prob, indices):
        condition = (indices[0] == indices[2])
        acc_prob += jax.lax.cond(condition, lambda _: p[tuple(indices)], lambda _: 0.0, None)
        return acc_prob, None

    prob_1 = 0
    prob_1, _ = jax.lax.scan(scan_fn, prob_1, generate_indices(p.shape))

    return -1.0 * (1 - prob_1) + 1.0 * prob_1

def exp_val_BC(p):
    def scan_fn(acc_prob, indices):
        condition = (indices[1] == indices[2])
        acc_prob += jax.lax.cond(condition, lambda _: p[tuple(indices)], lambda _: 0.0, None)
        return acc_prob, None

    prob_1 = 0
    prob_1, _ = jax.lax.scan(scan_fn, prob_1, generate_indices(p.shape))

    return -1.0 * (1 - prob_1) + 1.0 * prob_1

def exp_val_ABC(p):
    def scan_fn(acc_prob, indices):
        condition = (indices[0] + indices[1] + indices[2] == 1) | ((indices[0] == indices[1]) & (indices[0] == indices[2]) & (indices[1] == indices[2]) & (indices[0] == 1))
        acc_prob += jax.lax.cond(condition, lambda _: p[tuple(indices)], lambda _: 0.0, None)
        return acc_prob, None

    prob_1 = 0
    prob_1, _ = jax.lax.scan(scan_fn, prob_1, generate_indices(p.shape))

    return -1.0 * (1 - prob_1) + 1.0 * prob_1

def ineq_1(p):
    """
    0 <= 1 - <AC> - <BC> + <A><B>
    """
    return 1.0 - exp_val_AC(p) - exp_val_BC(p) + exp_val_A(p) * exp_val_B(p)

def ineq_2(p):
    """
    0 <= 3 - <A> - <B> - <C> + 2 * ( <AB> + <AC> + <BC> ) + <ABC> + <A><B> + <A><C> + <B><C> -
         - <A><BC> - <B><AC> - <C><AB> + <A><B><C>
    """
    return 3 - exp_val_A(p) - exp_val_B(p) - exp_val_C(p) + 2 * (exp_val_AB(p) + exp_val_AC(p) + exp_val_BC(p)) + exp_val_ABC(p) + exp_val_A(p) * exp_val_B(p) + exp_val_A(p) * exp_val_C(p) + exp_val_B(p) * exp_val_C(p) - exp_val_A(p) * exp_val_BC(p) - exp_val_B(p) * exp_val_AC(p) - exp_val_C(p) * exp_val_AB(p) + exp_val_A(p) * exp_val_B(p) * exp_val_C(p)

def ineq_3(p):
    """
    0 <= 4 + 2 * ( <C> - <AB> - <BC> + <A><B>) - 3 * <AC> - <ABC> + <A><C> - <A><BC> - <C><AB> + <A><B><C>
    """
    return 4 + 2 * (exp_val_C(p) - exp_val_AB(p) - exp_val_BC(p) + exp_val_A(p) * exp_val_B(p)) - 3 * exp_val_AC(p) - exp_val_ABC(p) + exp_val_A(p) * exp_val_C(p) - exp_val_A(p) * exp_val_BC(p) - exp_val_C(p) * exp_val_AB(p) + exp_val_A(p) * exp_val_B(p) * exp_val_C(p)
def ineq_4(p):
    """
    0 <= 4 + 2 ( - <AB> - <AC> - <BC> + <A><B> + <A><C> + <B><C> ) - <ABC> - <A><BC> - <B><AC> - <C><AB>
    """
    return 4 + 2 * ( - exp_val_AB(p) - exp_val_AC(p) - exp_val_BC(p) + exp_val_A(p) * exp_val_B(p) + exp_val_A(p) * exp_val_C(p) + exp_val_B(p) * exp_val_C(p)) - exp_val_ABC(p) - exp_val_A(p) * exp_val_BC(p) - exp_val_B(p) * exp_val_AC(p) - exp_val_C(p) * exp_val_AB(p)

if __name__ == "__main__":
    p = jnp.abs(np.random.uniform(size=(2, 2, 2, 1, 1, 1)))
    p = p / jnp.sum(p)
    print(f"<ABC> Matías: {ineq_4(p)}")
    print(f"<ABC> Cristian: {ineq4(p)}")

    print("Prob. distribution: ")
    p = np.array(p)
    for indices in np.ndindex(p.shape):
        value = p[indices]
        print(f"{indices} : {value}")