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
def step_jit(V, E_R, E_M, E_G, E_A, X, X_mods, T_e, T_e_mods, V_pending, V_satisfied, pred_ops, pred_cs, V_active, E_R_active, E_G_active, R_triggered, T_e_snapshot, X_snapshot):
    # Copy pending firing requests; they are set externally (e.g., by an RL agent)
    pending_flags = V_pending.copy()

    E_R_active[:]    = np.zeros(E_R.shape[0], dtype=np.bool_)

    # Single pass trigger evaluation
    V_targeted = np.zeros(V.shape[0], dtype=np.bool_)
    E_G_active[:] = np.zeros(E_G.shape[0], dtype=np.bool_)

    # TWO-PHASE EVALUATION to handle interactive actions before automatic triggers
    
    # PHASE 1: Only pending (interactive) nodes are active
    for i in range(V.shape[0]):
        V_active[i] = pending_flags[i]  # Only pending nodes active in phase 1

    # Evaluate triggers for pending actions first
    changed = True
    while changed:
        changed = False
        for i in range(E_G.shape[0]):
            src_id = int(E_G[i, 1])
            dest_id = int(E_G[i, 2])
            pred_id = int(E_G[i, 3])
            res_id = int(E_G[i, 4])
            dest_type = int(E_G[i, 5])
            trigger_mode = int(E_G[i, 6])  # 0=PASSIVE, 1=AUTOMATIC
            
            should_evaluate = False
            if trigger_mode == 0:  # PASSIVE only in phase 1
                fires_directly = V_active[src_id]
                propagates = V_targeted[src_id] and (V[src_id, 3] == 1)
                should_evaluate = fires_directly or propagates
            # Skip AUTOMATIC triggers in phase 1

            if not should_evaluate:
                continue

            # If there's a predicate, check it
            fires = True
            if pred_id != -1:
                fires = apply_pred(X[src_id, res_id], pred_ops[pred_id], pred_cs[pred_id])
            
            if not fires:
                continue

            # Mark active and target destination
            E_G_active[i] = True
            if dest_type == 1:  # ElementType.NODE
                if not V_targeted[dest_id]:
                    V_targeted[dest_id] = True
                    changed = True
                V_active[dest_id] = True
            elif dest_type == 2:  # ElementType.RESOURCE_CONNECTION
                R_triggered[dest_id] = True

    # PHASE 2: Add automatic nodes to active set
    for i in range(V.shape[0]):
        if V[i, 1] == 1:  # AUTOMATIC firing mode
            V_active[i] = True

    # Now evaluate automatic triggers with updated state
    changed = True
    while changed:
        changed = False
        for i in range(E_G.shape[0]):
            if E_G_active[i]:  # Skip already processed triggers
                continue
                
            src_id = int(E_G[i, 1])
            dest_id = int(E_G[i, 2])
            pred_id = int(E_G[i, 3])
            res_id = int(E_G[i, 4])
            dest_type = int(E_G[i, 5])
            trigger_mode = int(E_G[i, 6])  # 0=PASSIVE, 1=AUTOMATIC
            
            should_evaluate = False
            if trigger_mode == 1:  # AUTOMATIC
                should_evaluate = True
            else:  # PASSIVE (not yet processed)
                fires_directly = V_active[src_id]
                propagates = V_targeted[src_id] and (V[src_id, 3] == 1)
                should_evaluate = fires_directly or propagates

            if not should_evaluate:
                continue

            # If there's a predicate, check it
            fires = True
            if pred_id != -1:
                fires = apply_pred(X[src_id, res_id], pred_ops[pred_id], pred_cs[pred_id])
            
            if not fires:
                continue

            # Mark active and target destination
            E_G_active[i] = True
            if dest_type == 1:  # ElementType.NODE
                if not V_targeted[dest_id]:
                    V_targeted[dest_id] = True
                    changed = True
                V_active[dest_id] = True
            elif dest_type == 2:  # ElementType.RESOURCE_CONNECTION
                R_triggered[dest_id] = True

    # Resource connections 
    E_R_satisfied = np.zeros(E_R.shape[0], dtype=np.bool_)

    # Roll back node-targeting modifier contributions from previous step
    X -= X_mods
    
    # Zero-out modifier buffers; fresh values will be computed in two stages.
    X_mods[:] = np.zeros(X.shape, dtype=np.float64)
    T_e_mods[:] = np.zeros(T_e.shape, dtype=np.float64)

    # Stage-1 modifiers: those that change NODE values (must happen first)
    for i in range(E_M.shape[0]):
        dst_type      = int(E_M[i, 6])
        if dst_type != 1:
            continue  # node-targeting only in stage-1

        src_id       = int(E_M[i, 1])
        dst_id       = int(E_M[i, 2])
        src_res_id   = int(E_M[i, 3])
        dst_res_id   = int(E_M[i, 4])
        mod_rate_coef = E_M[i, 5]

        modifier_value = X[src_id, src_res_id] * mod_rate_coef
        X_mods[dst_id, dst_res_id] += modifier_value

    # Apply node-targeting modifier deltas
    X += X_mods

    # NOW evaluate activators AFTER modifiers are applied
    V_blocked = np.zeros(V.shape[0], dtype=np.bool_)
    R_blocked = np.zeros((1024,), dtype=np.bool_)
    for i in range(E_A.shape[0]):
        edge_id = int(E_A[i, 0])
        src_id     = int(E_A[i, 1])
        dest_id    = int(E_A[i, 2])
        pred_id     = int(E_A[i, 3])
        res_id     = int(E_A[i, 4])
        dst_type = int(E_A[i, 5])
        
        test_val = X[src_id, res_id]
        pred_result = apply_pred(test_val, pred_ops[pred_id], pred_cs[pred_id])
        
        if not pred_result:
            if dst_type == 1:
                V_blocked[dest_id] = True
            else:
                R_blocked[dest_id] = True

    # Update V_active based on blocking
    for i in range(V.shape[0]):
        if V_blocked[i]:
            V_active[i] = False

    # Update R_triggered based on blocking
    for i in range(len(R_triggered)):
        if R_blocked[i]:
            R_triggered[i] = False

    # Stage-2 modifiers: those that change RESOURCE_CONNECTION rates (after nodes updated)
    for i in range(E_M.shape[0]):
        dst_type      = int(E_M[i, 6])
        if dst_type != 2:
            continue  # skip node-targeting mods, already done

        src_id       = int(E_M[i, 1])
        dst_id       = int(E_M[i, 2])  # index into E_R array
        src_res_id   = int(E_M[i, 3])
        mod_rate_coef = E_M[i, 5]

        T_e_mods[dst_id] += X[src_id, src_res_id] * mod_rate_coef

    # Apply the connection-rate updates so the pulls below see them
    T_e += T_e_mods
    
    # Capture both X and T_e values for renderer (after all modifiers applied, before transfers)
    X_snapshot[:, :] = X[:, :]
    T_e_snapshot[:] = T_e[:]

    # Process resource connections
    for i in range(E_R.shape[0]):
        src_id = int(E_R[i, 1])
        dest_id = int(E_R[i, 2])

        # Skip if source node is blocked (but allow receiving resources)
        if V_blocked[src_id]:
            continue

        # Skip if activator blocked this edge
        if R_blocked[i]:
            continue

        # If the edge was explicitly triggered, it can fire regardless of
        # source-node activity; otherwise it needs the source active.
        # Note: i is the correct index since R_triggered was set using dest_id from trigger
        if R_triggered[i]:
            E_R_active[i] = True
        elif V_active[src_id]:
            E_R_active[i] = True

        if not E_R_active[i]:
            continue

        res_idx = int(E_R[i, 3])
        amount  = T_e[i]
        allow_partial = bool(E_R[i, 5])  # Get the allow_partial flag
        
        available = X[src_id, res_idx]
        
        if available >= amount:
            # Full transfer possible
            X[src_id, res_idx]  -= amount
            X[dest_id, res_idx] += amount
            E_R_satisfied[i] = True
        elif allow_partial and available > 0:
            # Partial transfer
            X[src_id, res_idx]  = 0  # Transfer everything available
            X[dest_id, res_idx] += available
            E_R_satisfied[i] = True  # Mark as satisfied since we transferred what we could

    # Roll back T_e modifiers AFTER all transfers
    T_e -= T_e_mods

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
    
    # Clear triggered resource connections for next iteration
    R_triggered[:] = np.zeros(len(R_triggered), dtype=np.bool_)

    return
