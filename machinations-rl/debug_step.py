from monopoly import *
import numpy as np

print("=== TESTING ACTIVATOR BLOCKING ===")

# Set player to have exactly 3 estates to test the boundary
m.X[player.id, estate.id] = 3.0
m.X[player.id, money.id] = 100.0

print(f"Initial: Estates={m.X[player.id, estate.id]}, Money={m.X[player.id, money.id]}")

# Check predicate manually
estate_count = m.X[player.id, estate.id]
pred_result = estate_count <= 3.0
should_block = not pred_result
print(f"Estate count: {estate_count}")
print(f"Predicate (estate <= 3): {pred_result}")
print(f"Should block: {should_block}")

# Try to buy
m.V_pending[buy.id] = True
print(f"Set buy node pending: {m.V_pending[buy.id]}")

# Check R_blocked before step
print(f"R_blocked before step: {m.R_triggered}")

m.step()

print(f"After step: Estates={m.X[player.id, estate.id]}, Money={m.X[player.id, money.id]}")

# Now test with 4 estates (should definitely block)
print("\n=== TESTING WITH 4 ESTATES ===")
m.X[player.id, estate.id] = 4.0
m.X[player.id, money.id] = 100.0

print(f"Initial: Estates={m.X[player.id, estate.id]}, Money={m.X[player.id, money.id]}")

# Check predicate manually
estate_count = m.X[player.id, estate.id]
pred_result = estate_count <= 3.0
should_block = not pred_result
print(f"Estate count: {estate_count}")
print(f"Predicate (estate <= 3): {pred_result}")
print(f"Should block: {should_block}")

# Try to buy
m.V_pending[buy.id] = True
m.step()

print(f"After step: Estates={m.X[player.id, estate.id]}, Money={m.X[player.id, money.id]}")
estate_gained = m.X[player.id, estate.id] - 4.0
print(f"Estate gained: {estate_gained} (should be 0 if blocked)") 