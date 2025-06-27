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

    # Approximate "node fires" condition **before** triggers affect V_active
    node_pre_fire = np.zeros(V.shape[0], dtype=np.bool_)
    for idx in range(V.shape[0]):
        node_pre_fire[idx] = (V[idx, 1] == 1) or pending_flags[idx]

    # ------------------------------------------------------------------
    # PASS 1 – triggers whose *source* is an INTERACTIVE node
    # ------------------------------------------------------------------
    R_triggered_phase1  = np.zeros((1024,), dtype=np.bool_)
    for i in range(E_G.shape[0]):
        src_id  = int(E_G[i, 1])
        if V[src_id, 1] != 2:  # firing_mode == INTERACTIVE ?
            continue  # skip non-interactive sources in phase-1

        dest_id = int(E_G[i, 2])
        pred_id = int(E_G[i, 3])
        res_id  = int(E_G[i, 4])
        dest_type = int(E_G[i, 5])

        # predicate check
        fires = True
        if pred_id != -1:
            fires = apply_pred(X[src_id, res_id], pred_ops[pred_id], pred_cs[pred_id])
        else:
            fires = node_pre_fire[src_id]

        if not fires:
            continue

        # Mark active
        E_G_active[i] = True
        if dest_type == 1:
            V_targeted[dest_id] = True
        elif dest_type == 2:
            R_triggered_phase1[dest_id] = not R_blocked[dest_id]

    # ------------------------------------------------------------------
    # EARLY resource-connection pull for edges triggered in phase-1
    # ------------------------------------------------------------------
    R_pulled = np.zeros((1024,), dtype=np.bool_)
    for i in range(E_R.shape[0]):
        if not R_triggered_phase1[i]:
            continue

        src_id  = int(E_R[i, 1])
        dest_id = int(E_R[i, 2])
        res_idx = int(E_R[i, 3])
        amount  = T_e[i]

        if X[src_id, res_idx] >= amount:
            X[src_id, res_idx]  -= amount
            X[dest_id, res_idx] += amount
            E_R_active[i] = True
            R_pulled[i]   = True

    # ------------------------------------------------------------------
    # PASS 2 – evaluate *all* triggers (including interactive again; ok)
    # ------------------------------------------------------------------
    for i in range(E_G.shape[0]):
        src_id  = int(E_G[i, 1])
        dest_id = int(E_G[i, 2])
        pred_id = int(E_G[i, 3])
        res_id = int(E_G[i, 4])
        dest_type = int(E_G[i, 5])

        if pred_id != -1:
            fires = apply_pred(X[src_id, res_id], pred_ops[pred_id], pred_cs[pred_id])
        else:
            fires = node_pre_fire[src_id]

        if fires:
            E_G_active[i] = True
            if dest_type == 1:
                V_targeted[dest_id] = True
            elif dest_type == 2:
                R_triggered[dest_id] = not R_blocked[dest_id]

    for i in range(V.shape[0]):
        V_active[i] = (V[i, 1] == 1 or pending_flags[i] or V_targeted[i]) and not V_blocked[i]

    # ------------------------------------------------------------
    # NEW PASS 3 – triggers originating from nodes that are now
    # active (because they fired this tick or were targeted by
    # earlier triggers).  This ensures that *any* firing node
    # propagates its outgoing triggers within the same tick.
    # Only triggers **without** an explicit predicate are handled
    # here – predicate-based triggers were already evaluated above.
    # ------------------------------------------------------------
    for i in range(E_G.shape[0]):
        # Skip triggers already marked active or those guarded by a predicate
        if E_G_active[i]:
            continue
        if int(E_G[i, 3]) != -1:
            continue  # predicate-based triggers already handled

        src_id  = int(E_G[i, 1])
        if not V_active[src_id]:
            continue  # source did not fire this tick

        dest_id   = int(E_G[i, 2])
        dest_type = int(E_G[i, 5])

        # Activate trigger
        E_G_active[i] = True
        if dest_type == 1:  # ElementType.NODE
            V_targeted[dest_id] = True
        elif dest_type == 2:  # ElementType.RESOURCE_CONNECTION
            R_triggered[dest_id] = not R_blocked[dest_id]

    # Recompute V_active to incorporate any nodes that became
    # targeted by the newly fired triggers above.
    for i in range(V.shape[0]):
        V_active[i] = (V[i, 1] == 1 or pending_flags[i] or V_targeted[i]) and not V_blocked[i]

    # Roll back previous modifier contributions so we can recompute them
    X -= X_mods
    T_e -= T_e_mods
    # Zero-out modifier buffers; fresh values will be computed in two stages.
    X_mods[:] = np.zeros(X.shape, dtype=np.float64)
    T_e_mods[:] = np.zeros(T_e.shape, dtype=np.float64)

    # ------------------------------------------------------------
    # Stage-1 modifiers: those that change RESOURCE_CONNECTION rates
    # (dst_type == 2) must be applied *before* we process the
    # resource-connection pulls in this tick.
    # ------------------------------------------------------------
    for i in range(E_M.shape[0]):
        dst_type      = int(E_M[i, 6])
        if dst_type != 2:
            continue  # skip node-targeting mods for now

        src_id       = int(E_M[i, 1])
        dst_id       = int(E_M[i, 2])  # index into E_R
        src_res_id   = int(E_M[i, 3])
        mod_rate_coef = E_M[i, 5]

        T_e_mods[dst_id] += X[src_id, src_res_id] * mod_rate_coef

    # Apply the connection-rate updates so the pulls below see them
    T_e += T_e_mods
    # DO NOT clear T_e_mods – we must keep them so the next tick can
    # roll them back correctly.

    # Resource connections 
    E_R_satisfied    = np.zeros(E_R.shape[0], dtype=np.bool_)

    for i in range(E_R.shape[0]):
        # Correct field positions: 1 = src.id, 2 = dst.id
        src_id  = int(E_R[i, 1])
        dest_id = int(E_R[i, 2])

        # Skip if already executed in phase-1
        if R_pulled[i]:
            continue

        # Skip if activator blocked this edge
        if R_blocked[i]:
            continue

        # If the edge was explicitly triggered, it can fire regardless of
        # source-node activity; otherwise it needs the source active.
        if (not R_triggered[i]) and (not V_active[src_id]):
            continue

        E_R_active[i] = True

        res_idx = int(E_R[i, 3])
        amount  = T_e[i]

        if X[src_id, res_idx] >= amount:
            X[src_id, res_idx]  -= amount
            X[dest_id, res_idx] += amount
            E_R_satisfied[i] = True

    # ------------------------------------------------------------
    # Stage-2 modifiers: those that change NODE_TARGETING rates
    # (dst_type == 1) must be applied *after* we process the
    # resource-connection pulls in this tick.
    # ------------------------------------------------------------
    for i in range(E_M.shape[0]):
        dst_type      = int(E_M[i, 6])
        if dst_type != 1:
            continue  # node-targeting only in stage-2

        src_id       = int(E_M[i, 1])
        dst_id       = int(E_M[i, 2])
        src_res_id   = int(E_M[i, 3])
        dst_res_id   = int(E_M[i, 4])
        mod_rate_coef = E_M[i, 5]

        X_mods[dst_id, dst_res_id] += X[src_id, src_res_id] * mod_rate_coef

    # Apply node-targeting modifier deltas
    X += X_mods

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
