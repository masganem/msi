import numpy as np
from .step_jit import step_jit
from .definitions import *

class Machinations:
    def __init__(
        self,
        resources,
        nodes,
        connections,
        resource_connections,
        label_modifiers,
        node_modifiers,
        triggers,
        activators,
        V,
        E,
        X,
        X_mods,
        T_e,
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
        self.label_modifiers = label_modifiers
        self.node_modifiers = node_modifiers
        self.triggers = triggers
        self.activators = activators
        self.V = V
        self.E_R, self.E_T, self.E_N, self.E_G, self.E_A = E
        self.X = X
        self.X_mods = X_mods
        self.T_e = T_e
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
                n.distribution_mode.value,
                0, # Placeholder to not mess anything up trust me
                n.quotient,
                n.resource_type.id if n.resource_type else -1
            ]
            for n in nodes
        ], dtype=float)

        for i, connection in enumerate(connections):
            connection.id = i

        # Split connections
        resource_connections = [
            c for c in connections if c.type == ElementType.RESOURCE_CONNECTION
        ]
        label_modifiers = [
            c for c in connections if c.type == ElementType.LABEL_MODIFIER
        ]
        node_modifiers = [
            c for c in connections if c.type == ElementType.NODE_MODIFIER
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
        # Columns: id, src, dst, resource_type, predicate_id, base_rate
        E_R = np.array([
            [
                c.id,
                c.src.id,
                c.dst.id,
                c.resource_type.id,
                c.predicate.id if c.predicate else -1,
                c.rate,  # store the initial/base rate here
            ]
            for c in resource_connections
        ], dtype=np.float64).reshape(-1, 6)

        # Label modifiers
        E_T = np.array([
            [
                c.id,
                c.src.id,
                c.dst.id,
                c.resource_type.id,
                c.rate,
            ]
            for c in label_modifiers
        ], dtype=np.float64).reshape(-1, 5)

        # Node modifiers
        E_N = np.array([
            [
                c.id,
                c.src.id,
                c.dst.id,
                c.src_resource_type.id,
                c.dst_resource_type.id,
                c.rate,
            ]
            for c in node_modifiers
        ], dtype=np.float64).reshape(-1, 6)

        # Triggers
        E_G = np.array([
            [
                c.id,
                c.src.id,
                c.dst.id,
                c.predicate.id if c.predicate else -1,
                c.weight,
                c.dst_type.value,
            ]
            for c in triggers
        ], dtype=np.float64).reshape(-1, 6)

        # Activators
        E_A = np.array([
            [
                c.id,
                c.src.id,
                c.dst.id,
                c.predicate.id,
                c.resource_type.id,
            ]
            for c in activators
        ], dtype=np.float64).reshape(-1, 5)

        # ------------------------------------------------------------
        # Build the initial state matrix *X*.
        #   • Start with explicitly supplied initial resources.
        #   • Then, for every NONDETERMINISTIC gate that did **not** specify an
        #     initial amount, draw a random value so that the gate does not
        #     start at zero by default.
        # ------------------------------------------------------------
        X = np.zeros((len(nodes), len(resources)), dtype=np.float64)

        # 1) User-supplied initial resources.
        for node in nodes:
            for resource, amount in node.initial_resources:
                X[node.id, resource.id] = float(amount)

        # 2) Autogenerated initial values for nondeterministic gates.
        rng = np.random.default_rng()
        for node in nodes:
            if node.type == ElementType.GATE and node.distribution_mode == DistributionMode.NONDETERMINISTIC:
                res_idx = node.resource_type.id
                # Skip if the user already provided a value for this resource.
                if X[node.id, res_idx] != 0.0:
                    continue

                if getattr(node, "values", None):
                    X[node.id, res_idx] = float(rng.choice(node.values))
                else:
                    X[node.id, res_idx] = float(rng.integers(node.quotient + 1))

        X_mods = np.zeros((len(nodes), len(resources)), dtype=np.float64)

        # Resource connection rates are part of the state
        T_e = np.zeros((len(resource_connections),), dtype=np.float64)
        for i, connection in enumerate(resource_connections):
            T_e[i] = float(connection.rate)

        return cls(
            resources,
            nodes,
            connections,
            resource_connections,
            label_modifiers,
            node_modifiers,
            triggers,
            activators,
            V,
            (E_R, E_T, E_N, E_G, E_A),
            X,
            X_mods,
            T_e,
            np.zeros(V.shape[0], dtype=np.bool_),
            np.zeros(V.shape[0], dtype=np.bool_),
            pred_ops,
            pred_cs,
            np.zeros(V.shape[0], dtype=np.bool_),
            np.zeros(E_R.shape[0], dtype=np.bool_),
            np.zeros(E_G.shape[0], dtype=np.bool_),
        )

    def step(self):
        """Advance the simulation by one tick.

        Prior to calling the low-level JIT kernel, sample a fresh value for
        every *NONDETERMINISTIC* gate.  If the gate defines ``.values`` (new
        API) we draw uniformly from that list; otherwise we fall back to the
        legacy behaviour of sampling `randint(0, quotient)`.
        """

        # ------------------------------------------------------------
        # 1. Sample nondeterministic gates (pure Python / NumPy).
        # ------------------------------------------------------------
        for node in self.nodes:
            if node.type == ElementType.GATE and node.distribution_mode == DistributionMode.NONDETERMINISTIC:
                res_idx = node.resource_type.id
                if getattr(node, "values", None):
                    # Uniform choice from explicit value list.
                    choice_idx = np.random.randint(len(node.values))
                    self.X[node.id, res_idx] = float(node.values[choice_idx])
                else:
                    # Back-compat: sample integer in [0, quotient].
                    self.X[node.id, res_idx] = float(np.random.randint(node.quotient + 1))

        # ------------------------------------------------------------
        # 2. Run the vectorised simulation kernel.
        # ------------------------------------------------------------
        step_jit(
            self.V, self.E_R, self.E_T, self.E_N, self.E_G,
            self.E_A, self.X, self.X_mods, self.T_e, self.V_pending,
            self.V_satisfied, self.pred_ops, self.pred_cs, self.V_active,
            self.E_R_active, self.E_G_active
        )
        self.t += 1

