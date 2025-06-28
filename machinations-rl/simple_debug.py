from monopoly import *

print("=== SIMPLE 2-STEP DEBUG ===")

# Set up: player has 4 estates (should block), and manually put money in buy node
m.X[player.id, estate.id] = 4.0
m.X[player.id, money.id] = 100.0
m.X[buy.id, money.id] = 20.0  # Pre-existing money in buy node

print(f"Initial: Player estates={m.X[player.id, estate.id]}, Player money={m.X[player.id, money.id]}, Buy money={m.X[buy.id, money.id]}")

# Step 1: Don't set buy node pending - just run step to see if modifier still runs
print("\n--- Step 1: No buy action, just step ---")
m.step()
print(f"After step 1: Player estates={m.X[player.id, estate.id]}, Player money={m.X[player.id, money.id]}, Buy money={m.X[buy.id, money.id]}")

# Step 2: Set buy node pending to see activator block
print("\n--- Step 2: With buy action ---")
m.V_pending[buy.id] = True
m.step()
print(f"After step 2: Player estates={m.X[player.id, estate.id]}, Player money={m.X[player.id, money.id]}, Buy money={m.X[buy.id, money.id]}") 