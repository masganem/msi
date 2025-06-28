from monopoly import *
import numpy as np

print("=== DETAILED ACTIVATOR DEBUG ===")

# Set up the scenario where blocking should happen
m.X[player.id, estate.id] = 4.0
m.X[player.id, money.id] = 100.0

# CRITICAL: Check and clear buy node money
print(f"Buy node money BEFORE clearing: {m.X[buy.id, money.id]}")
m.X[buy.id, money.id] = 0.0  # Clear any accumulated money
print(f"Buy node money AFTER clearing: {m.X[buy.id, money.id]}")

print(f"Player estates: {m.X[player.id, estate.id]}")
print(f"Player money: {m.X[player.id, money.id]}")

# Manually simulate the activator evaluation from step_jit.py
print("\n=== MANUAL ACTIVATOR SIMULATION ===")

# Get the activator data
E_A = m.E_A
X = m.X
pred_ops = m.pred_ops
pred_cs = m.pred_cs

print(f"E_A shape: {E_A.shape}")
if E_A.shape[0] > 0:
    i = 0  # First (and only) activator
    edge_id = int(E_A[i, 0])
    src_id = int(E_A[i, 1])
    dest_id = int(E_A[i, 2])
    pred_id = int(E_A[i, 3])
    res_id = int(E_A[i, 4])
    dst_type = int(E_A[i, 5])
    
    print(f"Activator {i}:")
    print(f"  edge_id: {edge_id}")
    print(f"  src_id: {src_id}")
    print(f"  dest_id: {dest_id}")
    print(f"  pred_id: {pred_id}")
    print(f"  res_id: {res_id}")
    print(f"  dst_type: {dst_type}")
    
    # Get the value to test
    test_value = X[src_id, res_id]
    print(f"  test_value (X[{src_id}, {res_id}]): {test_value}")
    
    # Get predicate info
    op_code = pred_ops[pred_id]
    c = pred_cs[pred_id]
    print(f"  predicate: op_code={op_code}, c={c}")
    
    # Apply predicate manually
    # op_code 2 is "<="
    if op_code == 2:
        pred_result = test_value <= c
    else:
        pred_result = False
        
    print(f"  predicate result ({test_value} <= {c}): {pred_result}")
    print(f"  should block (not pred_result): {not pred_result}")
    
    if not pred_result:
        print(f"  -> Would set R_blocked[{dest_id}] = True")
    else:
        print(f"  -> Would NOT block")

# Check the resource connection mapping
print(f"\n=== RESOURCE CONNECTION CHECK ===")
for i, conn in enumerate(m.resource_connections):
    print(f"RC[{i}]: id={conn.id}")
    if conn == player_buys_estate:
        print(f"  ^ player_buys_estate is at index {i}")

# Now actually run a step and see what happens
print(f"\n=== RUNNING ACTUAL STEP ===")
m.V_pending[buy.id] = True
print(f"Before step: estates={m.X[player.id, estate.id]}, money={m.X[player.id, money.id]}, buy_money={m.X[buy.id, money.id]}")
m.step()
print(f"After step: estates={m.X[player.id, estate.id]}, money={m.X[player.id, money.id]}, buy_money={m.X[buy.id, money.id]}") 