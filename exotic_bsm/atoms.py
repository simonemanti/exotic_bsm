"""Minimal exotic-atom model in natural units (hbar = c = 1)."""

from dataclasses import dataclass

import numpy as np
from particle import Particle

from .constants import ALPHA, EV_PER_MEV

ORBITING_PARTICLE_ALIASES = {
    "kaon": "K-",
    "pion": "pi-",
    "muon": "mu-",
    "antiproton": "p~-",
}


@dataclass(frozen=True)
class ExoticAtom:
    """Hydrogen-like atom formed by a nucleus and a negative particle."""

    Z: int
    A: int
    orbiting: str | int = "kaon"
    nucleus_mass_ev_override: float | None = None

    def __post_init__(self) -> None:
        if self.Z <= 0:
            raise ValueError("Z must be positive")
        if self.A < self.Z:
            raise ValueError("A must be greater than or equal to Z")
        if (
            self.nucleus_mass_ev_override is not None
            and self.nucleus_mass_ev_override <= 0
        ):
            raise ValueError("nucleus_mass_ev_override must be positive")

        self._validate_particle(self.orbiting_particle, "orbiting particle")
        self._validate_particle(self.nucleus_particle, "nucleus")

    @property
    def nucleus_pdgid(self) -> int:
        """PDG Monte Carlo code for the ground-state nucleus."""
        return 1_000_000_000 + self.Z * 10_000 + self.A * 10

    @property
    def orbiting_particle(self) -> Particle:
        """Negative particle orbiting the nucleus."""
        if isinstance(self.orbiting, str):
            particle_name = ORBITING_PARTICLE_ALIASES.get(
                self.orbiting.strip().lower(),
                self.orbiting.strip(),
            )
            particle = Particle.from_name(particle_name)
        else:
            particle = Particle.from_pdgid(self.orbiting)

        if particle.charge != -1:
            raise ValueError("the orbiting particle must have charge -1")
        return particle

    @property
    def nucleus_particle(self) -> Particle:
        """Ground-state nucleus identified from ``Z`` and ``A``."""
        return Particle.from_pdgid(self.nucleus_pdgid)

    @property
    def orbiting_name(self) -> str:
        """Canonical particle name used by ``particle``."""
        return self.orbiting_particle.name

    @property
    def orbiting_mass_ev(self) -> float:
        """Mass of the orbiting particle, converted from MeV to eV."""
        mass_mev = self.orbiting_particle.mass
        if mass_mev is None:
            raise ValueError("orbiting particle has no tabulated mass")
        return float(mass_mev) * EV_PER_MEV

    @property
    def nucleus_mass_ev(self) -> float:
        """Nuclear mass in eV, optionally replaced by an explicit value."""
        if self.nucleus_mass_ev_override is not None:
            return self.nucleus_mass_ev_override

        mass_mev = self.nucleus_particle.mass
        if mass_mev is None:
            raise ValueError("nucleus has no tabulated mass")
        return float(mass_mev) * EV_PER_MEV

    @property
    def reduced_mass_ev(self) -> float:
        """Reduced mass of the exotic atom in eV."""
        orbiting_mass = self.orbiting_mass_ev
        return (
            orbiting_mass
            * self.nucleus_mass_ev
            / (orbiting_mass + self.nucleus_mass_ev)
        )

    def coulomb_energy_ev(self, n: int) -> float:
        """Leading nonrelativistic Coulomb energy of level ``n`` in eV."""
        self._validate_n(n)
        return -self.reduced_mass_ev * (self.Z * ALPHA) ** 2 / (2 * n**2)

    def yukawa_element_ev(self, n: int, mediator_mass_ev):
        """Return ``<exp(-m_X r) / r>`` for a circular state in eV."""
        self._validate_n(n)
        mediator_mass = np.asarray(mediator_mass_ev, dtype=float)
        if np.any(mediator_mass < 0):
            raise ValueError("mediator masses cannot be negative")

        inverse_radius_ev = self.Z * ALPHA * self.reduced_mass_ev / n**2
        return inverse_radius_ev * (
            1 + mediator_mass / (2 * n * inverse_radius_ev)
        ) ** (-2 * n)

    @staticmethod
    def _validate_n(n: int) -> None:
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")

    @staticmethod
    def _validate_particle(particle: Particle, role: str) -> None:
        if particle.mass is None:
            raise ValueError(f"{role} has no tabulated mass")
