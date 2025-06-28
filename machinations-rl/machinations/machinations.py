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
        # Flag indicating the model has reached a terminal outcome (WIN/TIE/LOSE)
        self.terminated = False

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
                n.propagate.value,
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
        E_R = np.array([c.pack() for c in resource_connections], dtype=np.float64).reshape(-1, 6)

        # Modifiers
        E_M = np.array([c.pack() for c in modifiers], dtype=np.float64).reshape(-1, 7)

        # --------------------------------------------------------------------
        # For modifiers targeting resource connections, dst_id currently holds
        # the *connection id* (global among all connections).  The JIT kernel
        # expects the index into the E_R array.  Build a mapping and remap.
        # --------------------------------------------------------------------
        id_to_ridx = {int(E_R[i, 0]): i for i in range(E_R.shape[0])}
        for i in range(E_M.shape[0]):
            dst_type = int(E_M[i, 6])
            if dst_type == ElementType.RESOURCE_CONNECTION.value:
                conn_id = int(E_M[i, 2])
                E_M[i, 2] = id_to_ridx[conn_id]

        # Triggers
        E_G = np.array([c.pack() for c in triggers], dtype=np.float64).reshape(-1, 7)

        # Remap trigger destination ids when they point to resource connections
        for i in range(E_G.shape[0]):
            dst_type = int(E_G[i, 5])
            if dst_type == ElementType.RESOURCE_CONNECTION.value:
                conn_id = int(E_G[i, 2])
                if conn_id in id_to_ridx:
                    E_G[i, 2] = id_to_ridx[conn_id]

        # Activators
        E_A = np.array([c.pack() for c in activators], dtype=np.float64).reshape(-1, 6)

        # Remap activators that target resource-connections so dst id is the
        # index in E_R (needed by the JIT kernel for R_blocked logic).
        if E_A.size:
            for i in range(E_A.shape[0]):
                dst_type = int(E_A[i, 5])
                if dst_type == ElementType.RESOURCE_CONNECTION.value:
                    conn_id = int(E_A[i, 2])
                    if conn_id in id_to_ridx:
                        E_A[i, 2] = id_to_ridx[conn_id]

        X = np.zeros((len(nodes), len(resources)), dtype=np.float64)

        for node in nodes:
            if not node.distributions:
                continue
            for distribution in node.distributions:
                res_idx = distribution.resource_type.id
                v = distribution.draw(rng)
                if v is not None:
                    X[node.id, res_idx] = v

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
            np.zeros(T_e.shape[0], dtype=np.float64),
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

        # Skip further simulation once a terminal outcome was reached.
        if self.terminated:
            return

        for node in self.nodes:
            if not node.distributions:
                continue
            # Sampling rule: AUTOMATIC and NONDETERMINISTIC nodes sample every
            # tick; others sample only when the node is (about to be) active.
            if not (self.V_active[node.id] or node.firing_mode in (FiringMode.AUTOMATIC, FiringMode.NONDETERMINISTIC)):
                continue
            for distribution in node.distributions:
                res_idx = distribution.resource_type.id
                v = distribution.draw(rng)
                if v is not None:
                    self.X[node.id, res_idx] = v

        # ------------------------------------------------------------
        # Record a snapshot *before* the transfer kernel so the renderer can
        # show freshly drawn random values (e.g., deck contents) ahead of the
        # actual resource movement.
        # ------------------------------------------------------------
        if hasattr(self, "_renderer") and getattr(self, "_renderer") is not None:
            self._renderer._record_snapshot(extra={"phase": "pre"})

        # ------------------------------------------------------------
        # Predict connection rates **after** Stage-1 modifiers so they can be
        # displayed ahead of the actual transfer.  This uses the same formula
        # as the JIT kernel: new_rate = baseline + Σ src_val * coef for every
        # Modifier that targets a RESOURCE_CONNECTION.
        # ------------------------------------------------------------
        if hasattr(self, "_renderer") and getattr(self, "_renderer") is not None:
            import numpy as _np

            # Baseline = current T_e minus the modifier deltas that are still
            # stored in T_e_mods (they will be rolled back at the start of the
            # kernel).
            baseline = self.T_e - self.T_e_mods
            predicted = baseline.copy()

            for i in range(self.E_M.shape[0]):
                dst_type = int(self.E_M[i, 6])
                if dst_type != ElementType.RESOURCE_CONNECTION.value:
                    continue  # only connection-targeting mods

                src_id     = int(self.E_M[i, 1])
                dst_id     = int(self.E_M[i, 2])  # index into E_R array
                src_res_id = int(self.E_M[i, 3])
                coef       = self.E_M[i, 5]

                predicted[dst_id] += self.X[src_id, src_res_id] * coef

            self._renderer._record_snapshot(extra={"phase": "mid", "T_e": predicted})

        # 2. Run the simulation kernel.
        step_jit(
            self.V, self.E_R, self.E_M, self.E_G,
            self.E_A, self.X, self.X_mods, self.T_e, self.T_e_mods, self.V_pending,
            self.V_satisfied, self.pred_ops, self.pred_cs, self.V_active,
            self.E_R_active, self.E_G_active
        )
        self.t += 1

        # Check for OutcomeNode activation; if any fired, mark simulation terminated.
        for node in self.nodes:
            if isinstance(node, OutcomeNode) and self.V_active[node.id]:
                self.terminated = True
                break

