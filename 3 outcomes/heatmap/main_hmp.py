import sys
sys.path.append('..')

import numpy as np
import h5py
import jax.numpy as jnp
import jax
import parametrize
from tqdm import tqdm

# extracting data
data = np.genfromtxt('/home/matias/PycharmProjects/Triangle/3 '
                     'outcomes/heatmap/triangle-nonlocality-symmetric-subspace/data/symmetric_subspace_maps'
                     '/sym_subspace_map_card_3.csv', delimiter=',', skip_header=1)
threshold = 1/3
filtered_rows = data[data[:, 0] > threshold]
extracted_data = filtered_rows[:, :3]
presets = jnp.array(extracted_data)

def serialize_preset(preset):
    # Create a string representation with rounded precision to avoid floating-point representation issues
    return '_'.join(f"{v:.4f}" for v in preset)

# random parameters using JAX
def random_params(key):
    key, _ = jax.random.split(key)

    return jax.random.uniform(key, (132,)), key # we now need 132 parameters instead of 108
def loss_fn(params, target_p):
    return jnp.sum((parametrize.params_to_prob(params) - target_p) ** 2)

def elegant(preset):
    prob = np.zeros((3, 3, 3, 1, 1, 1))
    for a, b, c in np.ndindex(3, 3, 3):
        if a == b == c:
            prob[a, b, c, 0, 0, 0] = preset[0] / 3.0
        elif (a == b and a != c) or (b == c and b != a) or (a == c and a != b):
            prob[a, b, c, 0, 0, 0] = preset[1] / 18.0
        elif a != b and a != c and b != c:
            prob[a, b, c, 0, 0, 0] = preset[2] / 6.0

    return jnp.array(prob)

# looping code
if __name__ == "__main__":
    key = jax.random.PRNGKey(5)
    lr_initial = 1
    lr_final = 1e-1
    epochs = 10000
    loss_jit = jax.jit(loss_fn)
    grad_ls = jax.jit(jax.grad(loss_fn))
    loss_ls = []

    with h5py.File('params.h5', 'w') as f:
        for row in range(presets.shape[0]):
            params, key = random_params(key)
            print(f'Preset {presets[row]}')
            for epoch in tqdm(range(epochs)):
                if epoch < 8000:
                    params = params - lr_initial * grad_ls(params, elegant(presets[row]))
                else:
                    params = params - lr_final * grad_ls(params, elegant(presets[row]))
                if epoch % 500 == 0:
                    print(f"Epoch {epoch}: loss = {loss_jit(params, elegant(presets[row]))}")

            dataset_name = serialize_preset(presets[row])
            f.create_dataset(dataset_name, data=params)
            loss_ls.append(loss_jit(params, elegant(presets[row])))

    with open('quantum_distance.csv', 'w') as f:
        f.write('s111,s112,s123,euclidean\n')

        for row in range(presets.shape[0]):
            row_data = presets[row, :3]
            loss = loss_ls[row]
            f.write(','.join(map(str, row_data)) + ',' + str(loss) + '\n')