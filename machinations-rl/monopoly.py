from machinations.gym_env import MachinationsEnv
from machinations import Machinations
from machinations.definitions import *
from math import inf

money = Resource("Money")
estate = Resource("Estate")
luck = Resource("Luck")

rent_values = [*range(2, 52, 2), 25, 25, 25, 25, 28, 28]
rent_distribution = Distribution(luck, rent_values)

# Nodes
player = Node(FiringMode.PASSIVE, initial_resources=[(money, 500),(estate,0)])
bank = Node(FiringMode.PASSIVE, initial_resources=[(money, inf)])
world = Node(FiringMode.PASSIVE, initial_resources=[(money, inf)])
buy = Node(FiringMode.INTERACTIVE, initial_resources=[(money, 0)])
pass_go_odds = Node(FiringMode.NONDETERMINISTIC, distributions=[Distribution(luck, range(1, 41))])
pay_rent_odds = Node(FiringMode.NONDETERMINISTIC, distributions=[Distribution(luck, range(1, 41))])
get_rent_odds = Node(FiringMode.NONDETERMINISTIC, distributions=[Distribution(luck, range(1, 41))])
random_rent_value1 = Node(FiringMode.NONDETERMINISTIC, distributions=[rent_distribution])

# Edges
player_gets_go = ResourceConnection(bank, player, money, 50)
pass_go = Trigger(pass_go_odds, player_gets_go, luck, Predicate("<=", 7), TriggerMode.AUTOMATIC)
player_pays_rent = ResourceConnection(player, world, money, 0)
how_much_rent = Modifier(random_rent_value1, player_pays_rent, luck, money, 1)
pay_rent = Trigger(pay_rent_odds, player_pays_rent, luck, Predicate("<=", 28), TriggerMode.AUTOMATIC)
player_buys_estate = ResourceConnection(player, buy, money, 100)
buy_trigger = Trigger(buy, player_buys_estate)
estate_modifier = Modifier(buy, player, money, estate, 1/100)
player_gets_rent = ResourceConnection(world, player, money, 0)
estate_rent_modifier = Modifier(player, player_gets_rent, estate, money, 20)
get_rent = Trigger(get_rent_odds, player_gets_rent, luck, Predicate("<=", 7), TriggerMode.AUTOMATIC)
likelier_to_get_rent = Modifier(player, get_rent_odds, estate, luck, -1)

m = Machinations.load((
    [player, bank, world, buy, pass_go_odds, pay_rent_odds, random_rent_value1, get_rent_odds],
    [player_gets_go, pass_go, player_pays_rent, how_much_rent, pay_rent, player_buys_estate, buy_trigger, estate_modifier, player_gets_rent, estate_rent_modifier, get_rent, likelier_to_get_rent],
    [money, estate, luck],
))
