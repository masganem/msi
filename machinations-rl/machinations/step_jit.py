from numba import njit
import numpy as np

@njit
def random_choice(lst):
    idx = np.random.randint(0, len(lst))
    return lst[idx]

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
def step_jit(V, E_R, E_M, E_G, E_A, X, X_mods, T_e, T_e_mods, V_pending, V_satisfied, pred_ops, pred_cs, V_active, E_R_active, E_G_active):
    # Copy pending firing requests; they are set externally (e.g., by an RL agent)
    pending_flags = V_pending.copy()

    E_R_active[:]    = np.zeros(E_R.shape[0], dtype=np.bool_)

    # Evaluate activators
    V_blocked = np.zeros(V.shape[0], dtype=np.bool_)
    R_blocked = np.zeros((1024,), dtype=np.bool_)
    for i in range(E_A.shape[0]):
        edge_id = int(E_A[i, 0])
        src_id     = int(E_A[i, 1])
        dest_id    = int(E_A[i, 2])
        pred_id     = int(E_A[i, 3])
        res_id     = int(E_A[i, 4])
        dst_type = int(E_A[i, 5])
        if not apply_pred(X[src_id, res_id], pred_ops[pred_id], pred_cs[pred_id]):
            if dst_type == 1:
                V_blocked[dest_id] = True
            else:
                R_blocked[dest_id] = True

    # Evaluate triggers
    V_targeted   = np.zeros(V.shape[0], dtype=np.bool_)
    E_G_active[:] = np.zeros(E_G.shape[0], dtype=np.bool_)
    # Edges (indices in E_R) that are triggered this tick
    R_triggered  = np.zeros((1024,), dtype=np.bool_)

    # Pass 1: triggers 
    for i in range(E_G.shape[0]):
        src_id  = int(E_G[i, 1])
        dest_id = int(E_G[i, 2])
        pred_id = int(E_G[i, 3])
        res_id = int(E_G[i, 4])
        dest_type = int(E_G[i, 5])

        if (pred_id == -1 and V_satisfied[src_id]) or apply_pred(X[src_id, res_id], pred_ops[pred_id], pred_cs[pred_id]):
            E_G_active[i] = True
            if dest_type == 1:
                V_targeted[dest_id] = True
            elif dest_type == 2:
                R_triggered[dest_id] = not R_blocked[dest_id]

    for i in range(V.shape[0]):
        V_active[i] = (V[i, 1] == 1 or pending_flags[i] or V_targeted[i]) and not V_blocked[i]

    # Run modifiers
    X -= X_mods
    T_e -= T_e_mods
    X_mods[:] = np.zeros(X.shape, dtype=np.float64)
    T_e_mods[:] = np.zeros(T_e.shape, dtype=np.float64)
    for i in range(E_M.shape[0]):
        src_id       = int(E_M[i, 1])
        dst_id  = int(E_M[i, 2])  
        src_res_id       = int(E_M[i, 3])
        dst_res_id       = int(E_M[i, 4])
        mod_rate_coef = E_M[i, 5]
        dst_type = int(E_M[i, 6])
        if dest_type == 1:
            X_mods[dst_id, dst_res_id] = X[src_id, src_res_id] * mod_rate_coef
        elif dest_type == 2:
            T_e_mods[dst_id] = X[src_id, src_res_id] * mod_rate_coef
    X += X_mods
    T_e += T_e_mods

    # Resource connections 
    E_R_satisfied    = np.zeros(E_R.shape[0], dtype=np.bool_)

    for i in range(E_R.shape[0]):
        src_id  = int(E_R[i, 2])

        if (not V_active[src_id]) and (not R_triggered[i]):
            continue  

        E_R_active[i] = True

        dest_id  = int(E_R[i, 2])
        res_idx = int(E_R[i, 3])
        amount  = T_e[i]

        if X[src_id, res_idx] >= amount:
            X[src_id, res_idx]  -= amount
            X[dest_id, res_idx] += amount
            E_R_satisfied[i] = True

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

    # Clear pending flags for next iteration
    V_pending[:] = np.zeros(V.shape[0], dtype=np.bool_)

    return
