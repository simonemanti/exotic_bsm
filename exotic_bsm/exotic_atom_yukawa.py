"""Nonrelativistic Yukawa sensitivities for circular exotic-atom states."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


ALPHA = 1 / 137.035999177


@dataclass(frozen=True)
class ExoticAtomYukawa:
    """Hydrogen-like exotic atom in natural units (hbar = c = 1)."""

    Z: int
    mass_orbiting_ev: float
    mass_nucleus_ev: float

    def __post_init__(self) -> None:
        if self.Z <= 0:
            raise ValueError("Z must be positive")
        if self.mass_orbiting_ev <= 0 or self.mass_nucleus_ev <= 0:
            raise ValueError("particle masses must be positive")

    @property
    def reduced_mass_ev(self) -> float:
        """Reduced mass in eV."""
        return (
            self.mass_orbiting_ev
            * self.mass_nucleus_ev
            / (self.mass_orbiting_ev + self.mass_nucleus_ev)
        )

    def coulomb_energy_ev(self, n: int) -> float:
        """Leading nonrelativistic Coulomb energy of level n, in eV."""
        self._validate_n(n)
        return -self.reduced_mass_ev * (self.Z * ALPHA) ** 2 / (2 * n**2)

    def yukawa_element_ev(self, n: int, mediator_mass_ev):
        """Return h_n^X = <exp(-m_X r)/r> in eV."""
        self._validate_n(n)
        mediator_mass_ev = np.asarray(mediator_mass_ev, dtype=float)
        if np.any(mediator_mass_ev < 0):
            raise ValueError("mediator masses cannot be negative")

        inverse_radius_ev = self.Z * ALPHA * self.reduced_mass_ev / n**2
        return inverse_radius_ev * (
            1 + mediator_mass_ev / (2 * n * inverse_radius_ev)
        ) ** (-2 * n)

    def transition_energy_ev(self, n_initial: int, n_final: int) -> float:
        """Positive leading-order transition energy, in eV."""
        self._validate_transition(n_initial, n_final)
        return abs(
            self.coulomb_energy_ev(n_initial)
            - self.coulomb_energy_ev(n_final)
        )

    def transition_yukawa_ev(
        self,
        n_initial: int,
        n_final: int,
        mediator_mass_ev,
    ):
        """Yukawa matrix-element difference for a transition, in eV."""
        self._validate_transition(n_initial, n_final)
        return self.yukawa_element_ev(
            n_initial, mediator_mass_ev
        ) - self.yukawa_element_ev(n_final, mediator_mass_ev)

    def coupling_limit(
        self,
        n_initial: int,
        n_final: int,
        mediator_mass_ev,
        precision_ppm: float,
        confidence_multiplier: float = 2.0,
    ):
        """Projected limit on |g_H^X g_N^X| for a relative precision."""
        if precision_ppm <= 0:
            raise ValueError("precision_ppm must be positive")
        if confidence_multiplier <= 0:
            raise ValueError("confidence_multiplier must be positive")

        sigma_ev = self.transition_energy_ev(n_initial, n_final) * precision_ppm * 1e-6
        shift_ev = self.transition_yukawa_ev(
            n_initial,
            n_final,
            mediator_mass_ev,
        )
        return (
            confidence_multiplier
            * 4
            * np.pi
            * sigma_ev
            / np.abs(shift_ev)
        )

    def plot_sensitivity_curve(
        self,
        n_initial: int,
        n_final: int,
        precision_ppm: float,
        mass_min_kev: float = 1.0,
        mass_max_kev: float = 1e5,
        points: int = 500,
        confidence_multiplier: float = 2.0,
        ax=None,
        **plot_kwargs,
    ):
        """Plot the projected sensitivity curve and return ``(figure, axes)``."""
        if mass_min_kev <= 0 or mass_max_kev <= mass_min_kev:
            raise ValueError("require 0 < mass_min_kev < mass_max_kev")
        if points < 2:
            raise ValueError("points must be at least 2")

        if ax is None:
            figure, ax = plt.subplots(figsize=(7.2, 5.0))
        else:
            figure = ax.figure

        mediator_mass_kev = np.logspace(
            np.log10(mass_min_kev),
            np.log10(mass_max_kev),
            points,
        )
        limits = self.coupling_limit(
            n_initial,
            n_final,
            mediator_mass_kev * 1e3,
            precision_ppm,
            confidence_multiplier,
        )

        plot_kwargs.setdefault(
            "label",
            rf"$n={n_initial}\to{n_final}$, {precision_ppm:g} ppm",
        )
        ax.loglog(mediator_mass_kev, limits, **plot_kwargs)
        ax.set_xlabel(r"$m_X$ [keV]")
        ax.set_ylabel(r"$|g_H^X g_N^X|$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        figure.tight_layout()
        plt.show()
        return figure, ax

    @staticmethod
    def _validate_n(n: int) -> None:
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")

    @classmethod
    def _validate_transition(cls, n_initial: int, n_final: int) -> None:
        cls._validate_n(n_initial)
        cls._validate_n(n_final)
        if n_initial <= n_final:
            raise ValueError("n_initial must be larger than n_final")
