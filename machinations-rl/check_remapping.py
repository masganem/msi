from monopoly import *
import numpy as np

print("=== CHECKING ACTIVATOR REMAPPING ===")

# Check the mapping from connection ID to resource connection index
print("Resource connections:")
id_to_ridx = {}
for i, conn in enumerate(m.resource_connections):
    id_to_ridx[conn.id] = i
    print(f"  RC[{i}]: id={conn.id}")
    if conn == player_buys_estate:
        print(f"    ^ player_buys_estate")

print(f"\nid_to_ridx mapping: {id_to_ridx}")

# Check the activator before and after remapping
print(f"\nActivator details:")
print(f"  limit_estate.dst.id: {limit_estate.dst.id}")
print(f"  limit_estate.dst_type: {limit_estate.dst_type}")

# Check the E_A array
print(f"\nE_A array:")
print(f"  E_A shape: {m.E_A.shape}")
if m.E_A.shape[0] > 0:
    print(f"  E_A[0]: {m.E_A[0]}")
    dst_id = int(m.E_A[0, 2])
    dst_type = int(m.E_A[0, 5])
    print(f"  dst_id: {dst_id}")
    print(f"  dst_type: {dst_type}")
    
    if dst_type == 2:  # RESOURCE_CONNECTION
        print(f"  Expected mapping: {limit_estate.dst.id} -> {id_to_ridx.get(limit_estate.dst.id, 'NOT FOUND')}")
        if dst_id == id_to_ridx.get(limit_estate.dst.id):
            print("  ✓ Remapping is correct")
        else:
            print("  ✗ Remapping is WRONG") 