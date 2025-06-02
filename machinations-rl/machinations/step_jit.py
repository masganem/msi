from numba import jit
import numpy as np

@jit
def step_jit(V, E_R, E_T, E_N, E_G, E_A, X, T_e):
    # Find active nodes
    V_active = np.empty(V.shape[0], dtype=np.bool_)
    for i in range(V.shape[0]):
        # Node is automatic
        V_active[i] = (V[i, 1] == 1)
        # OR Interactive node was activated last turn
        # OR Node is the target of a trigger whose source node had all its inputs satisfied (jesus christ)
        # AND Node is not blocked by activators

    # Run random gates

    # Find active resource edges
    E_R_active = np.empty(E_R.shape[0], dtype=np.bool_)
    for i in range(E_R.shape[0]):
        E_R_active[i] = V_active[int(E_R[i, 2])]

    # Transfer resources
    for i in range(E_R.shape[0]):
        if E_R_active[i]:
            edge_id = int(E_R[i, 0])
            dest    = int(E_R[i, 1])
            src     = int(E_R[i, 2])
            res_idx = int(E_R[i, 3])
            amount = T_e[edge_id]
            X[src,  res_idx] += amount
            X[dest, res_idx] -= amount

    # Update resource edge rates

    # Update node states per node modifiers

    return X, T_e
