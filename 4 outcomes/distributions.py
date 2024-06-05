import qutip as qt
import numpy as np
import jax.numpy as jnp

def P_Fritz(visibility):
    def qreshape(inp, dims):
        return qt.Qobj(inp.data.to_array(), dims=dims)

    # CHSH measurements

    A0 = qt.sigmax()
    A1 = qt.sigmaz()
    A = [[A0.eigenstates()[1][0].proj(),
          A0.eigenstates()[1][1].proj()],
         [A1.eigenstates()[1][0].proj(),
          A1.eigenstates()[1][1].proj()]]

    B0 = (-qt.sigmax() - qt.sigmaz()) / np.sqrt(2)
    B1 = (-qt.sigmax() + qt.sigmaz()) / np.sqrt(2)
    B = [[B0.eigenstates()[1][0].proj(),
          B0.eigenstates()[1][1].proj()],
         [B1.eigenstates()[1][0].proj(),
          B1.eigenstates()[1][1].proj()]]

    # define state

    b0 = qt.ket("0")
    b1 = qt.ket("1")
    b00 = qt.tensor(b0, b0)
    b01 = qt.tensor(b0, b1)
    b10 = qt.tensor(b1, b0)
    b11 = qt.tensor(b1, b1)

    def state_noise(visibility):
        Id4 = qreshape(0.25 * qt.identity(4), [[2, 2], [2, 2]])
        # What I will use as the shared state between A and B.
        Phi_plus = 1 / np.sqrt(2) * (b01 - b10)  # 1/2**0.5 * (01-10)
        ini_state = qt.ket2dm(Phi_plus)
        return visibility * ini_state + (1 - visibility) * (Id4)

    ################### Set noise here #######################################
    rho_AB = state_noise(visibility)

    # the following two are 0,5*|00><00|+0,5*|11><11|
    rho_BC = 0.5 * qt.ket2dm(b00) + 0.5 * qt.ket2dm(b11)
    rho_CA = 0.5 * qt.ket2dm(b00) + 0.5 * qt.ket2dm(b11)

    # The following line of code gives the state in tensor product of the form
    # A1 B1 B2 C1 C2 A2
    # 1  2  3  4  5  6
    # where
    #  A --(A1)-- rho_AB --(B1)-- B
    #    \                       /
    #     (A2)                (B2)
    #        \                /
    #       rho_CA        rho_BC
    #           \          /
    #           (C2)    (C1)
    #              \    /
    #                C
    # where between parenthesis we have Hilbert spaces of dimension 2
    # ex B2  === \mathcal{H}_{B2}=\mathbb{C}^2

    full_state_ini = qt.tensor(rho_AB, rho_BC, rho_CA)

    # Now for the measurements it is more convenient to use the ordering
    # A2 A1 B1 B2 C1 C2, so we need to change to this.
    # 6  1  2  3  4  5
    #
    # End(A1 B1 B2 C1 C2 A2) = Sum (..) |i>_1 |j>_2 |k>_3 |l>_4 |m>_5 |n>_6
    #                                   <o|_1 <p|_2 <q|_3 <r|_4 <s|_5 <t|_6
    #                             |
    #                             |
    #                             V
    # End(A2 A1 B1 B2 C1 C2) = Sum (..) |n>_6 |i>_1 |j>_2 |k>_3 |l>_4 |m>_5
    #                                   <t|_6 <o|_1 <p|_2 <q|_3 <r|_4 <s|_5
    # and this corresponds to shifting all the corresponding indices.
    # full_state = np.reshape(full_state, [2,2,  2,2,  2,2,  2,2,  2,2,  2,2])
    #                             idx nr  0 1   2 3   4 5   6 7   8 9  10 11
    full_state = qt.tensor_swap(full_state_ini, [4, 5])
    full_state = qt.tensor_swap(full_state, [3, 4])
    full_state = qt.tensor_swap(full_state, [2, 3])
    full_state = qt.tensor_swap(full_state, [1, 2])
    full_state = qt.tensor_swap(full_state, [0, 1])

    full_state = qt.tensor_swap(full_state, [10, 11])
    full_state = qt.tensor_swap(full_state, [9, 10])
    full_state = qt.tensor_swap(full_state, [8, 9])
    full_state = qt.tensor_swap(full_state, [7, 8])
    full_state = qt.tensor_swap(full_state, [6, 7])

    # define measurements for each party in terms of CHSH measurements

    b0proj = b0.proj()
    b1proj = b1.proj()

    A00 = qt.tensor(b0proj, A[0][0])
    A01 = qt.tensor(b0proj, A[0][1])
    A10 = qt.tensor(b1proj, A[1][0])
    A11 = qt.tensor(b1proj, A[1][1])

    Ameas = [A00, A01, A10, A11]

    B00 = qt.tensor(B[0][0], b0proj)
    B01 = qt.tensor(B[0][1], b0proj)
    B10 = qt.tensor(B[1][0], b1proj)
    B11 = qt.tensor(B[1][1], b1proj)

    Bmeas = [B00, B01, B10, B11]

    C00 = qt.tensor(b0proj, b0proj)
    C10 = qt.tensor(b0proj, b1proj)
    C01 = qt.tensor(b1proj, b0proj)
    C11 = qt.tensor(b1proj, b1proj)

    Cmeas = [C00, C01, C10, C11]

    # Build probability

    prob = np.zeros((4, 4, 4, 1, 1, 1))
    for a, b, c in np.ndindex(4, 4, 4):
        prob[a, b, c, 0, 0, 0] = (full_state * qt.tensor(Ameas[a], Bmeas[b], Cmeas[c])).tr().real

    return jnp.array(prob)

def elegant():
    prob = np.zeros((4, 4, 4, 1, 1, 1))
    for a, b, c in np.ndindex(4, 4, 4):
        if a == b == c:
            prob[a, b, c, 0, 0, 0] = 25/256
        elif (a == b and a != c) or (b == c and b != a) or (a == c and a != b):
            prob[a, b, c, 0, 0, 0] = 1/256
        elif a != b and a != c and b != c:
            prob[a, b, c, 0, 0, 0] = 5/256

    return jnp.array(prob)


if __name__ == '__main__':
    print(jnp.sum(elegant()))