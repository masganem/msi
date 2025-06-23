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
    E_G_active[:] = np.zeros(E_G.shape[0], dtype=np.bool_)

    # Pass 1: triggers originating from pools
    for i in range(E_G.shape[0]):
        src  = int(E_G[i, 1])
        dest = int(E_G[i, 2])

        # Pools have V[src, 2] == 0 (see ElementType.POOL)
        if V[src, 2] == 0:
            if V_satisfied[src]:
                E_G_active[i] = True
                V_targeted[dest] = True

    for i in range(V.shape[0]):
        # Gate
        if V[i, 2] == 1:
            # Nondeterministic
            if V[i, 3] == 1:
                # Remember gates can only have one resource type
                res_idx = int(V[i, 6])
                X[i, res_idx] = randint(0, int(V[i, 5]))
            
            # Triggers leaving from this gate
            for j in range(E_G.shape[0]):
                if int(E_G[j, 1]) != int(V[i, 0]):
                    continue  # Not coming from this gate

                dst_id   = int(E_G[j, 2])
                pred_idx = int(E_G[j, 3])

                # Gates refer to their own single resource type stored in V[i, 6]
                res_idx = int(V[i, 6])

                # Fire trigger if predicate holds (or no predicate defined)
                if pred_idx == -1 or apply_pred(X[i, res_idx], pred_ops[pred_idx], pred_cs[pred_idx]):
                    E_G_active[j] = True
                    V_targeted[dst_id] = True

            # Resource connections leaving from gate

        # Compute active nodes for the current iteration (will be recomputed for all later)
        V_active[i] = (V[i, 1] == 1 or V_pending[i] or V_targeted[i]) and not V_blocked[i]

    # Re-evaluate active state for every node now that all triggers have been processed.
    for i in range(V.shape[0]):
        V_active[i] = (V[i, 1] == 1 or V_pending[i] or V_targeted[i]) and not V_blocked[i]

    # ------------------------------------------------------------
    # Resource connections – two-phase evaluation
    #   1. Edges WITHOUT predicate: require destination firing.
    #   2. Edges WITH    predicate: evaluated AFTER phase-1 transfers, so
    #      newly received resources can immediately enable subsequent flows.
    # ------------------------------------------------------------

    E_R_active[:]    = np.zeros(E_R.shape[0], dtype=np.bool_)
    E_R_satisfied    = np.zeros(E_R.shape[0], dtype=np.bool_)

    # Phase 1 – no-predicate edges (dest must be active)
    for i in range(E_R.shape[0]):
        pred_id  = int(E_R[i, 4])
        if pred_id != -1:
            continue  # handled in phase-2

        dest_id  = int(E_R[i, 2])
        if not V_active[dest_id]:
            continue  # destination not firing – cannot transfer this edge

        src_id  = int(E_R[i, 1])
        res_idx = int(E_R[i, 3])
        amount  = T_e[i]

        E_R_active[i] = True
        if X[src_id, res_idx] >= amount:
            X[src_id, res_idx]  -= amount
            X[dest_id, res_idx] += amount
            E_R_satisfied[i] = True

    # Phase 2 – predicate edges (destination activity irrelevant)
    for i in range(E_R.shape[0]):
        pred_id  = int(E_R[i, 4])
        if pred_id == -1:
            continue  # already handled in phase-1

        src_id  = int(E_R[i, 1])
        dest_id = int(E_R[i, 2])
        res_idx = int(E_R[i, 3])

        if not apply_pred(X[src_id, res_idx], pred_ops[pred_id], pred_cs[pred_id]):
            continue  # predicate failed – no transfer

        amount = T_e[i]

        E_R_active[i] = True
        if X[src_id, res_idx] >= amount:
            X[src_id, res_idx]  -= amount
            X[dest_id, res_idx] += amount
            E_R_satisfied[i] = True
        # else: not enough resource – E_R_satisfied remains False

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
