from machinations import Machinations

class Renderer:
    """
    Simulation recorder for Machinations models.
    Collects model snapshots and provides hooks for external visualization.
    """
    def __init__(self, model: Machinations):
        """
        Initialize with a Machinations model and record its initial state.
        """
        self.model = model
        self.history: list[dict] = []
        self._record_snapshot()

    def _record_snapshot(self, extra: dict | None = None) -> None:
        """Record a snapshot of the current simulation state.

        Parameters
        ----------
        extra : dict, optional
            Additional key/value pairs to store in the snapshot (e.g. actions
            taken by an agent or *V_pending* flags). The supplied dictionary is
            shallow-copied before insertion so that later mutations do not
            affect the stored history.
        """
        snap = {
            't': self.model.t,
            'X': self.model.X.copy(),
            'T_e': self.model.T_e.copy(),
            'V_active': self.model.V_active.copy(),
            'V_pending': self.model.V_pending.copy(),
            'E_R_active': self.model.E_R_active.copy(),
            'E_G_active': self.model.E_G_active.copy(),
        }
        if extra:
            snap.update(extra.copy())
        self.history.append(snap)

    def simulate(self, steps: int) -> None:
        """
        Advance the model by `steps` iterations, recording each state internally.
        """
        for _ in range(steps):
            self.model.step()
            self._record_snapshot()

    def render(self, steps: int = 10) -> None:
        """
        Prepare history by simulating for `steps`.
        Actual rendering must be performed by a separate Manim Scene,
        which can import this Renderer instance or consume its `.history`.
        """
        self.simulate(steps)

