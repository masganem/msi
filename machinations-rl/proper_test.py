from monopoly import *

print("=== PROPER TEST FROM BEGINNING ===")

# Start fresh - don't manually set anything
print(f"Initial: Player estates={m.X[player.id, estate.id]}, Player money={m.X[player.id, money.id]}, Buy money={m.X[buy.id, money.id]}")

# Let player buy estates until activator blocks
for step in range(8):
    print(f"\n--- Step {step + 1} ---")
    m.V_pending[buy.id] = True
    print(f"Before: Player estates={m.X[player.id, estate.id]}, Player money={m.X[player.id, money.id]}, Buy money={m.X[buy.id, money.id]}")
    
    m.step()
    
    print(f"After: Player estates={m.X[player.id, estate.id]}, Player money={m.X[player.id, money.id]}, Buy money={m.X[buy.id, money.id]}")
    
    if m.terminated:
        print("Game terminated")
        break 