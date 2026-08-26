"""Experimental inputs associated with exotic-atom transitions."""

from dataclasses import dataclass

import numpy as np

from .transitions import Transition


@dataclass(frozen=True)
class Measurement:
    """Experimental precision and optional central value for a transition."""

    transition: Transition
    stat_uncertainty_ev: float
    syst_uncertainty_ev: float
    energy_ev: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.stat_uncertainty_ev):
            raise ValueError("stat_uncertainty_ev must be finite")
        if not np.isfinite(self.syst_uncertainty_ev):
            raise ValueError("syst_uncertainty_ev must be finite")
        if self.stat_uncertainty_ev < 0:
            raise ValueError("stat_uncertainty_ev cannot be negative")
        if self.syst_uncertainty_ev < 0:
            raise ValueError("syst_uncertainty_ev cannot be negative")
        if self.total_uncertainty_ev == 0:
            raise ValueError("at least one experimental uncertainty must be positive")
        if self.energy_ev is not None and not np.isfinite(self.energy_ev):
            raise ValueError("energy_ev must be finite")
        if self.source is not None and not self.source.strip():
            raise ValueError("source cannot be empty")

    @property
    def total_uncertainty_ev(self) -> float:
        """Experimental uncertainty with stat and syst added in quadrature."""
        return float(
            np.hypot(
                self.stat_uncertainty_ev,
                self.syst_uncertainty_ev,
            )
        )
