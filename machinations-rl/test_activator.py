from monopoly import *

print('Initial estate:', m.X[player.id, estate.id])
print('Initial money:', m.X[player.id, money.id])

# Manually set pending to buy
for i in range(10):  # Extended to 10 steps
    print(f'\n--- Step {i+1} ---')
    m.V_pending[buy.id] = True
    print(f'Before step: Estates={m.X[player.id, estate.id]}, Money={m.X[player.id, money.id]}, Buy_money={m.X[buy.id, money.id]}')
    
    # Check if activator should block
    estate_count = m.X[player.id, estate.id]
    should_block = estate_count > 3
    print(f'Estate count: {estate_count}, Should block: {should_block}')
    
    m.step()
    estates_after = m.X[player.id, estate.id]
    money_after = m.X[player.id, money.id]
    buy_money_after = m.X[buy.id, money.id]
    print(f'After step: Estates={estates_after}, Money={money_after}, Buy_money={buy_money_after}')
    
    # Check if purchase actually happened
    estate_gained = estates_after - estate_count
    print(f'Estate gained: {estate_gained}')
    
    if m.terminated:
        print('Game terminated')
        break 