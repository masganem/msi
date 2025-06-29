from machinations import Machinations
from machinations.definitions import *
from math import inf

money = Resource("Money")
estate = Resource("Estate")
luck = Resource("Luck")

# REBALANCED: Increased pressure to invest while keeping it learnable
rent_values = np.array([*range(2, 8, 2), 10, 10, 12, 12, 15, 15])
print(rent_values)
rent_distribution = Distribution(luck, rent_values)

# Outcomes
lose_node = OutcomeNode("lose")

# Nodes
player = Node(FiringMode.PASSIVE, initial_resources=[(money, 50),(estate,0)])
bank = Node(FiringMode.PASSIVE, initial_resources=[(money, inf)])
world = Node(FiringMode.PASSIVE, initial_resources=[(money, inf)])
buy = Node(FiringMode.INTERACTIVE, initial_resources=[(money, 0)])
pass_go_odds = Node(FiringMode.NONDETERMINISTIC, distributions=[Distribution(luck, range(1, 41))])
# REBALANCED: Increased rent chance to create investment pressure (55%)
pay_rent_odds = Node(FiringMode.NONDETERMINISTIC, distributions=[Distribution(luck, range(1, 41))])
get_rent_odds = Node(FiringMode.NONDETERMINISTIC, distributions=[Distribution(luck, range(1, 41))])
random_rent_value1 = Node(FiringMode.NONDETERMINISTIC, distributions=[rent_distribution])

# Edges
# REBALANCED: Pass go income $8, good chance (25%)
player_gets_go = ResourceConnection(bank, player, money, 5)
pass_go = Trigger(pass_go_odds, player_gets_go, luck, Predicate("<=", 7), TriggerMode.AUTOMATIC)
player_pays_rent = ResourceConnection(player, world, money, 0, True)
how_much_rent = Modifier(random_rent_value1, player_pays_rent, luck, money, 1)
# REBALANCED: Increased rent chance to 55% - forces investment strategy
pay_rent = Trigger(pay_rent_odds, player_pays_rent, luck, Predicate("<=", 28), TriggerMode.AUTOMATIC)
player_buys_estate = ResourceConnection(player, buy, money, 5)
buy_trigger = Trigger(buy, player_buys_estate)
estate_modifier = Modifier(buy, player, money, estate, 1/5)
player_gets_rent = ResourceConnection(world, player, money, 0)
# REBALANCED: Estate rent $12 per estate - makes investment worthwhile
estate_rent_modifier = Modifier(player, player_gets_rent, estate, money, 4)
get_rent = Trigger(get_rent_odds, player_gets_rent, luck, Predicate("<=", 14), TriggerMode.AUTOMATIC)
likelier_to_get_rent = Modifier(player, get_rent_odds, estate, luck, -1)
lose = Trigger(player, lose_node, money, Predicate("<=", 0), TriggerMode.AUTOMATIC)
limit_estate = Activator(player, player_buys_estate, estate, Predicate("<", 15))

m = Machinations.load((
    [lose_node, player, bank, world, buy, pass_go_odds, pay_rent_odds, random_rent_value1, get_rent_odds],
    [player_gets_go, pass_go, player_pays_rent, how_much_rent, pay_rent, player_buys_estate, buy_trigger, estate_modifier, player_gets_rent, estate_rent_modifier, get_rent, likelier_to_get_rent, lose, limit_estate],
    [money, estate, luck],
))

# ------------------------------------------------------------
# Node positions and styling
# ------------------------------------------------------------
lose_node.pos = (0, 3)
bank.pos = (-3, 0)
world.pos = (3, 0)
buy.pos = (0, -3)
pass_go_odds.pos = (-1.5, 2)
pay_rent_odds.pos = (1.5, -2)
get_rent_odds.pos = (1.5, 2)
random_rent_value1.pos = (3, -2)

# Node aliases for legend
alias_map = {
    player.id: "Player",
    world.id: "World",
    bank.id: "Bank",
    pass_go_odds.id: "Odds of passing go",
    pay_rent_odds.id: "Odds of paying rent",
    get_rent_odds.id: "Odds of getting rent",
    random_rent_value1.id: "Random rent value", 
}

# Node and connection display names
for node in m.nodes:
    if isinstance(node, OutcomeNode):
        node.name = f"$\\text{{{node.outcome.upper()}}}$"
    else:
        node.name = f"$V_{{{node.id}}}$"
    
    if node.id in alias_map:
        setattr(node, "alias", alias_map[node.id])

buy.name = "$\\text{BUY}$"

for conn in m.connections:
    conn.name = f"$E_{{{conn.id}}}$"

# Resource colors
for resource in m.resources:
    if resource.name == "Money":
        resource.color = "#22fe22"
    elif resource.name == "Estate":
        resource.color = "#882222"
    else:
        resource.color = "#0f0f0f"
