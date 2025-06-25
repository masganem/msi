from numba import njit
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
def step_jit(V, E_R, E_T, E_N, E_G, E_A, X, X_mods, T_e, V_pending, V_satisfied, pred_ops, pred_cs, V_active, E_R_active, E_G_active):
    # Copy pending firing requests; they are set externally (e.g., by an RL agent)
    pending_flags = V_pending.copy()

    E_R_active[:]    = np.zeros(E_R.shape[0], dtype=np.bool_)

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
    V_targeted   = np.zeros(V.shape[0], dtype=np.bool_)
    E_G_active[:] = np.zeros(E_G.shape[0], dtype=np.bool_)
    # Edges (indices in E_R) that are triggered this tick
    R_triggered  = np.zeros(E_R.shape[0], dtype=np.bool_)

    # Pass 1: triggers originating from pools
    for i in range(E_G.shape[0]):
        src  = int(E_G[i, 1])
        dest = int(E_G[i, 2])
        dest_type = int(E_G[i, 5])

        # Pools have V[src, 2] == 0 (see ElementType.POOL)
        if V[src, 2] == 0:
            if V_satisfied[src]:
                E_G_active[i] = True
                if dest_type == 0 or dest_type == 1:
                    V_targeted[dest] = True
                elif dest_type == 2:
                    # Map connection id -> row index in E_R
                    for ridx in range(E_R.shape[0]):
                        if int(E_R[ridx, 0]) == dest:
                            R_triggered[ridx] = True
                            break

    for i in range(V.shape[0]):
        # Gate
        if V[i, 2] == 1:
            # NOTE: For nondeterministic gates the resource value (X[i, res_idx])
            #       is now sampled *outside* this JIT function, prior to calling
            #       `step_jit`.  The old in-place `randint` logic has been removed
            #       to allow arbitrary choice distributions.
            
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
                    dst_type = int(E_G[j, 5])
                    if dst_type == 0 or dst_type == 1:
                        V_targeted[dst_id] = True
                    elif dst_type == 2:
                        for ridx in range(E_R.shape[0]):
                            if int(E_R[ridx, 0]) == dst_id:
                                R_triggered[ridx] = True
                                break

            # Resource connections leaving from gate

        # Compute active nodes for the current iteration (will be recomputed for all later)
        V_active[i] = (V[i, 1] == 1 or pending_flags[i] or V_targeted[i]) and not V_blocked[i]

    # Re-evaluate active state for every node now that all triggers have been processed.
    for i in range(V.shape[0]):
        V_active[i] = (V[i, 1] == 1 or pending_flags[i] or V_targeted[i]) and not V_blocked[i]

    # -----------------------------
    # Update resource edge rates (EARLY)
    #   • Start from each edge's base rate (stored in E_R[:,5]).
    #   • Add contributions from label-modifier connections.
    #   This must execute BEFORE the resource-transfer phases so that the
    #   updated rates are used immediately in the current tick.
    # -----------------------------
    for ridx in range(E_R.shape[0]):
        T_e[ridx] = E_R[ridx, 5]

    for i in range(E_T.shape[0]):
        # Row fields: (id, src.id, dst_conn.id, resource_id, rate)
        src_id       = int(E_T[i, 1])
        dst_conn_id  = int(E_T[i, 2])  # connection ID of the target resource edge
        res_id       = int(E_T[i, 3])
        mod_rate_coef = E_T[i, 4]

        # Locate the corresponding resource-edge row index.
        for ridx in range(E_R.shape[0]):
            if int(E_R[ridx, 0]) == dst_conn_id:
                T_e[ridx] += mod_rate_coef * X[src_id, res_id]
                break

    # ------------------------------------------------------------
    # Resource connections – two-phase evaluation
    #   1. Edges WITHOUT predicate: require destination firing.
    #   2. Edges WITH    predicate: evaluated AFTER phase-1 transfers, so
    #      newly received resources can immediately enable subsequent flows.
    # ------------------------------------------------------------

    E_R_satisfied    = np.zeros(E_R.shape[0], dtype=np.bool_)

    # Phase 1 – no-predicate edges (dest must be active)
    for i in range(E_R.shape[0]):
        pred_id  = int(E_R[i, 4])
        if pred_id != -1:
            continue  # handled in phase-2

        dest_id  = int(E_R[i, 2])
        if (not V_active[dest_id]) and (not R_triggered[i]):
            continue  # cannot transfer this edge this phase
        if E_R_active[i]:
            continue # will deal with this later

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

    X -= X_mods
    X_mods[:] = np.zeros(X.shape, dtype=np.float64)
    # Update node states per node modifiers
    for i in range(E_N.shape[0]):
        # Row fields: (id, src.id, dst_conn.id, resource_id, rate)
        src_id       = int(E_N[i, 1])
        dst_id  = int(E_N[i, 2])  
        src_res_id       = int(E_N[i, 3])
        dst_res_id       = int(E_N[i, 4])
        mod_rate_coef = E_N[i, 5]
        X_mods[dst_id, dst_res_id] = X[src_id, src_res_id] * mod_rate_coef
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
