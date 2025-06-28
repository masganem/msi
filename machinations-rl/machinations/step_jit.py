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

    # Single pass trigger evaluation
    V_targeted = np.zeros(V.shape[0], dtype=np.bool_)
    E_G_active[:] = np.zeros(E_G.shape[0], dtype=np.bool_)

    # Initialize V_active based on firing modes and pending flags
    for i in range(V.shape[0]):
        V_active[i] = ((V[i, 1] == 1) or pending_flags[i]) and not V_blocked[i]

    # Keep evaluating triggers until no new nodes become active
    changed = True
    while changed:
        changed = False
        # For each trigger
        for i in range(E_G.shape[0]):
            src_id = int(E_G[i, 1])
            
            # Never evaluate triggers from blocked nodes
            if V_blocked[src_id]:
                continue
                
            dest_id = int(E_G[i, 2])
            pred_id = int(E_G[i, 3])
            res_id = int(E_G[i, 4])
            dest_type = int(E_G[i, 5])
            trigger_mode = int(E_G[i, 6])  # 0=PASSIVE, 1=AUTOMATIC
            
            should_evaluate = False
            if trigger_mode == 1:  # AUTOMATIC
                should_evaluate = True
            else:  # PASSIVE
                # For PASSIVE triggers, source node must either:
                # 1. Be firing (automatic/interactive/pending)
                # 2. Be targeted AND have CONTINUE propagation mode
                fires_directly = V_active[src_id]  # Already computed above
                propagates = V_targeted[src_id] and (V[src_id, 3] == 1)  # Check propagate mode
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
                if not V_targeted[dest_id]:  # Only mark changed if this is new
                    V_targeted[dest_id] = True
                    changed = True
                # Immediately update V_active if node isn't blocked
                if not V_blocked[dest_id]:
                    V_active[dest_id] = True
            elif dest_type == 2:  # ElementType.RESOURCE_CONNECTION
                # dest_id is already the index into E_R array, so R_triggered[dest_id]
                # will be checked later as R_triggered[i] in the resource connection loop
                R_triggered[dest_id] = not R_blocked[dest_id]

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

        X_mods[dst_id, dst_res_id] += X[src_id, src_res_id] * mod_rate_coef

    # Apply node-targeting modifier deltas
    X += X_mods

    # Stage-2 modifiers: those that change RESOURCE_CONNECTION rates (after nodes updated)
    for i in range(E_M.shape[0]):
        dst_type      = int(E_M[i, 6])
        if dst_type != 2:
            continue  # skip node-targeting mods, already done

        src_id       = int(E_M[i, 1])
        dst_id       = int(E_M[i, 2])  # index into E_R array
        src_res_id   = int(E_M[i, 3])
        mod_rate_coef = E_M[i, 5]

        print("Stage-2 Modifier:")
        print(i)
        print("src_id:")
        print(src_id)
        print("dst_id:")
        print(dst_id)
        print("src_res_id:")
        print(src_res_id)
        print("mod_rate_coef:")
        print(mod_rate_coef)
        print("X[src_id, src_res_id]:")
        print(X[src_id, src_res_id])
        print("T_e_mods[dst_id] before:")
        print(T_e_mods[dst_id])

        T_e_mods[dst_id] += X[src_id, src_res_id] * mod_rate_coef
        
        print("T_e_mods[dst_id] after:")
        print(T_e_mods[dst_id])

    print("DEBUG: Player X values:")
    print("X[1, 0] (money):")
    print(X[1, 0])
    print("X[1, 1] (estate):")
    print(X[1, 1])
    print("X[1, 2] (luck):")
    print(X[1, 2])

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
        print("Connection:")
        print(i)
        print("src:")
        print(src_id)
        print("dst:")
        print(dest_id)
        print("res:")
        print(res_idx)
        print("amount:")
        print(amount)
        print("available:")
        print(available)
        print("partial:")
        print(allow_partial)

        if available >= amount:
            print("FULL TRANSFER")
            # Full transfer possible
            X[src_id, res_idx]  -= amount
            X[dest_id, res_idx] += amount
            E_R_satisfied[i] = True
        elif allow_partial and available > 0:
            print("PARTIAL")
            # Partial transfer
            X[src_id, res_idx]  = 0  # Transfer everything available
            X[dest_id, res_idx] += available
            E_R_satisfied[i] = True  # Mark as satisfied since we transferred what we could
        else:
            print("NO TRANSFER")

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

    return
