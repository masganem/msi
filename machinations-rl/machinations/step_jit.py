from numba import njit
from random import randint
import numpy as np

@njit
def apply_pred(x: float, op_code: int, c: float) -> bool:
    # 0 ==, 1 <, 2 <=, 3 >, 4 >=, 5 !=
    if   op_code == 0: return x == c
    elif op_code == 1: return x  < c
    elif op_code == 2: return x <= c
    elif op_code == 3: return x  > c
    elif op_code == 4: return x >= c
    else:               return x != c

# TODO: jesus christ, this function signature... too ugly... 
@njit
def step_jit(V, E_R, E_T, E_N, E_G, E_A, X, T_e, V_pending, V_satisfied, pred_ops, pred_cs, V_active, E_R_active, E_G_active):
    # Placeholder
    V_pending = np.zeros(V.shape[0], dtype=np.bool_)

    # Evaluate activators
    V_blocked = np.zeros(V.shape[0], dtype=np.bool_)
    for i in range(E_A.shape[0]):
        edge_id = int(E_A[i, 0])
        src     = int(E_A[i, 1])
        dest    = int(E_A[i, 2])
        pred_id     = int(E_A[i, 3])
        res_idx     = int(E_A[i, 4])
        if not apply_pred(X[src, res_idx], pred_ops[pred_id], pred_cs[pred_id]):
            V_blocked[dest] = True

    # Evaluate triggers
    V_targeted = np.zeros(V.shape[0], dtype=np.bool_)
    for i in range(E_G.shape[0]):
        edge_id = int(E_G[i, 0])
        src     = int(E_G[i, 1])
        dest    = int(E_G[i, 2])
        if V_satisfied[src]:
            V_targeted[dest] = True
            E_G_active[i] = True
        else:
            V_targeted[dest] = False
            E_G_active[i] = False

    for i in range(V.shape[0]):
        V_active[i] = (V[i, 1] == 1 or V_pending[i] or V_targeted[i]) and not V_blocked[i]
        # Gate
        if V[i, 2] == 1:
            # Nondeterministic
            if V[i, 3] == 1:
                # TODO: Set random temporary state. How to think about quotient/weight here? Should I be doing quotient/weight? What random functions are usable in Numba?
                # Remember gates can only have one resource type
                res_idx = int(V[i, 6])
                X[i, res_idx] = randint(0, int(V[i, 5]))

            for j, e in enumerate(list(E_G)):
                src_id = int(e[1])
                dst_id = int(e[2])
                pred_id = int(e[4])
                if src_id == int(V[i, 0]):
                    if apply_pred(X[src_id, res_idx], pred_ops[pred_id], pred_cs[pred_id]):
                        V_active[dst_id] = True
                        E_G_active[j] = True

    # Find active resource edges
    for i in range(E_R.shape[0]):
        E_R_active[i] = V_active[int(E_R[i, 2])]

    # Transfer resources
    E_R_satisfied = np.ones(E_R.shape[0], dtype=np.bool_)
    for i in range(E_R.shape[0]):
        if E_R_active[i]:
            edge_id = int(E_R[i, 0])
            src     = int(E_R[i, 1])
            dest    = int(E_R[i, 2])
            res_idx = int(E_R[i, 3])
            amount = T_e[edge_id]
            if X[src, res_idx] < amount:
                E_R_satisfied[i] = False
                continue
            X[src,  res_idx] -= amount
            X[dest, res_idx] += amount
            E_R_satisfied[i] = True
        else:
            E_R_satisfied[i] = False

    V_satisfied[:] = np.ones(V.shape[0], dtype=np.bool_)

    # Track which nodes have at least one incoming resource connection
    has_incoming = np.zeros(V.shape[0], dtype=np.bool_)
    for j in range(E_R.shape[0]):
        dest = int(E_R[j, 2])
        has_incoming[dest] = True
        if not E_R_satisfied[j]:
            V_satisfied[dest] = False

    # Nodes with no incoming resource connections are considered unsatisfied
    for i in range(V.shape[0]):
        if not has_incoming[i]:
            V_satisfied[i] = False

    # Update resource edge rates

    # Update node states per node modifiers

    return
