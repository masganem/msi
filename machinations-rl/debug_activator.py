from monopoly import *
import numpy as np

print("=== ACTIVATOR DEBUG ===")
print(f"Activator: {limit_estate}")
print(f"Activator src: {limit_estate.src} (id: {limit_estate.src.id})")
print(f"Activator dst: {limit_estate.dst} (id: {limit_estate.dst.id})")
print(f"Activator resource: {limit_estate.resource_type} (id: {limit_estate.resource_type.id})")
print(f"Activator predicate: {limit_estate.predicate}")
print(f"Activator dst_type: {limit_estate.dst_type}")

print("\n=== RESOURCE CONNECTIONS ===")
for i, conn in enumerate(m.resource_connections):
    print(f"RC[{i}]: {conn} (id: {conn.id}) - {conn.src} -> {conn.dst}")
    if conn == player_buys_estate:
        print(f"  ^ This is player_buys_estate (index {i})")

print("\n=== ACTIVATOR ARRAY ===")
print(f"E_A shape: {m.E_A.shape}")
if m.E_A.shape[0] > 0:
    for i in range(m.E_A.shape[0]):
        print(f"E_A[{i}]: {m.E_A[i]}")
        print(f"  src_id: {int(m.E_A[i, 1])}")
        print(f"  dst_id: {int(m.E_A[i, 2])}")
        print(f"  pred_id: {int(m.E_A[i, 3])}")
        print(f"  res_id: {int(m.E_A[i, 4])}")
        print(f"  dst_type: {int(m.E_A[i, 5])}")

print("\n=== PREDICATE ARRAYS ===")
print(f"pred_ops: {m.pred_ops}")
print(f"pred_cs: {m.pred_cs}")

print("\n=== INITIAL STATE ===")
print(f"Player estates: {m.X[player.id, estate.id]}")
print(f"Player money: {m.X[player.id, money.id]}") 