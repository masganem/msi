from numba import jit
import numpy as np

# TODO: jesus christ, this function signature... too ugly... 
@jit
def step_jit(V, E_R, E_T, E_N, E_G, E_A, X, T_e, V_pending, V_satisfied):
    # Evaluate activators
    V_blocked = np.zeros(V.shape[0], dtype=np.bool_)
    for i in range(E_A.shape[0])
        edge_id = int(E_A[i, 0])
        dest    = int(E_A[i, 1])
        src     = int(E_A[i, 2])
        # TODO: Call predicate and block accordingly

    # Evaluate triggers
    V_targeted = np.zeros(V.shape[0], dtype=np.bool_)
    for i in range(E_G.shape[0])
        edge_id = int(E_G[i, 0])
        dest    = int(E_G[i, 1])
        src     = int(E_G[i, 2])
        if V_satisfied[src]:
            V[targeted] = True
    
    V_active = np.zeros(V.shape[0], dtype=np.bool_)
    for i in range(V.shape[0]):
        V_active[i] = (V[i, 1] == 1 or V_pending[i] or V_targeted[i]) and not V_blocked[i]
        # Gate
        if V[i, 2] == 2:
            # Nondeterministic
            if V[i, 3] == 1:
                # TODO: Set random temporary state. How to think about quotient/weight here? Should I be doing quotient/weight? What random functions are usable in Numba?
                # X[i, int(V[i, 5])] = ...

    # Run random gates

    # Find active resource edges
    E_R_active = np.zeros(E_R.shape[0], dtype=np.bool_)
    for i in range(E_R.shape[0]):
        E_R_active[i] = V_active[int(E_R[i, 2])]

    # Transfer resources
    V_satisfied = np.ones(V.shape[0], dtype=np.bool_)
    for i in range(E_R.shape[0]):
        if E_R_active[i]:
            # TODO: implement "fail if not enough resources in source"
            edge_id = int(E_R[i, 0])
            src     = int(E_R[i, 1])
            dest    = int(E_R[i, 2])
            res_idx = int(E_R[i, 3])
            amount = T_e[edge_id]
            if X[src, res_idx] < amount:
                V_satisfied[dest] = False
                continue
            X[src,  res_idx] -= amount
            X[dest, res_idx] += amount
        else:
            V_satisfied[dest] = False

    # Update resource edge rates

    # Update node states per node modifiers

    return X, T_e, V_pending, V_satisfied
