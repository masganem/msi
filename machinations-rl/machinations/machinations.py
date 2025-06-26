import numpy as np
from .step_jit import step_jit
from .definitions import *

rng = np.random.default_rng()


class Machinations:
    def __init__(
        self,
        resources,
        nodes,
        connections,
        resource_connections,
        modifiers,
        triggers,
        activators,
        V,
        E,
        X,
        X_mods,
        T_e,
        T_e_mods,
        V_pending,
        V_satisfied,
        pred_ops,
        pred_cs,
        V_active,
        E_R_active,
        E_G_active,
    ):
        self.t = 0
        self.resources = resources
        self.nodes = nodes
        self.connections = connections
        self.resource_connections = resource_connections
        self.modifiers = modifiers
        self.triggers = triggers
        self.activators = activators
        self.V = V
        self.E_R, self.E_M, self.E_G, self.E_A = E
        self.X = X
        self.X_mods = X_mods
        self.T_e = T_e
        self.T_e_mods = T_e_mods
        self.pred_ops = pred_ops
        self.pred_cs = pred_cs
        self.V_pending = V_pending
        self.V_satisfied = V_satisfied
        self.V_active = V_active
        self.E_R_active = E_R_active
        self.E_G_active = E_G_active

    @classmethod
    def load(cls, d: Diagram):
        # Load diagram definitions
        nodes, connections, resources = d

        for i, node in enumerate(nodes):
            node.id = i

        for i, resource in enumerate(resources):
            resource.id = i

        V = np.array([
            [
                n.id,
                n.firing_mode.value,
                n.type.value,
            ]
            for n in nodes
        ], dtype=np.float64)

        for i, connection in enumerate(connections):
            connection.id = i

        # Split connections
        resource_connections = [
            c for c in connections if c.type == ElementType.RESOURCE_CONNECTION
        ]

        modifiers = [
            c for c in connections if c.type == ElementType.MODIFIER
        ]

        triggers = [
            c for c in connections if c.type == ElementType.TRIGGER
        ]
        activators = [
            c for c in connections if c.type == ElementType.ACTIVATOR
        ]

        predicates = [c.predicate for c in connections if hasattr(c, "predicate") and c.predicate]
        for idx, p in enumerate(predicates):
            p.id = idx
        pred_ops = np.array([p.op_code for p in predicates], dtype=np.int8)
        pred_cs  = np.array([p.c       for p in predicates], dtype=np.float64)


        # Resource connections
        E_R = np.array([c.pack() for c in resource_connections], dtype=np.float64).reshape(-1, 5)

        # Modifiers
        E_M = np.array([c.pack() for c in modifiers], dtype=np.float64).reshape(-1, 7)

        # Triggers
        E_G = np.array([c.pack() for c in triggers], dtype=np.float64).reshape(-1, 6)

        # Activators
        E_A = np.array([c.pack() for c in activators], dtype=np.float64).reshape(-1, 6)

        X = np.zeros((len(nodes), len(resources)), dtype=np.float64)

        for node in nodes:
            if not node.distributions:
                continue
            for distribution in node.distributions:
                res_idx = distribution.resource_type.id
                X[node.id, res_idx] = float(rng.choice(distribution.values))

        for node in nodes:
            for resource, amount in node.initial_resources:
                X[node.id, resource.id] = float(amount)

        modifier_values = np.zeros((len(nodes), len(resources)), dtype=np.float64)

        # Resource connection rates are part of the state
        T_e = np.zeros((len(resource_connections),), dtype=np.float64)
        for i, connection in enumerate(resource_connections):
            T_e[i] = float(connection.rate)

        return cls(
            resources,
            nodes,
            connections,
            resource_connections,
            modifiers,
            triggers,
            activators,
            V,
            (E_R, E_M, E_G, E_A),
            X,
            modifier_values,
            T_e,
            np.zeros(T_e.shape[0], dtype=np.bool_),
            np.zeros(V.shape[0], dtype=np.bool_),
            np.zeros(V.shape[0], dtype=np.bool_),
            pred_ops,
            pred_cs,
            np.zeros(V.shape[0], dtype=np.bool_),
            np.zeros(E_R.shape[0], dtype=np.bool_),
            np.zeros(E_G.shape[0], dtype=np.bool_),
        )

    def step(self):
        """Advance the simulation by one tick."""

        for node in self.nodes:
            if not self.V_active[node.id]:
                continue
            if not node.distributions:
                continue
            for distribution in node.distributions:
                res_idx = distribution.resource_type.id
                self.X[node.id, res_idx] = float(rng.choice(distribution.values))

        # 2. Run the simulation kernel.
        step_jit(
            self.V, self.E_R, self.E_M, self.E_G,
            self.E_A, self.X, self.X_mods, self.T_e, self.T_e_mods, self.V_pending,
            self.V_satisfied, self.pred_ops, self.pred_cs, self.V_active,
            self.E_R_active, self.E_G_active
        )
        self.t += 1

