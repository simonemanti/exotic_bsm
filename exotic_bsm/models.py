"""Interaction models for exotic-atom energy shifts."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .constants import HIGGS_VEV_EV
from .transitions import Transition


class YukawaCoupling:
    """Generic Yukawa interaction parameterized by ``g_o * g_A``.

    The potential convention is

        V(r) = -(g_o * g_A) exp(-m_X r) / (4 pi r).

    Consequently, ``shift_coefficient_ev`` returns the coefficient ``K`` in
    ``delta_E = K * (g_o * g_A)``.
    """

    name: ClassVar[str] = "Yukawa coupling"
    parameter_label: ClassVar[str] = r"$|g_H^X g_N^X|$"
    parameter_unit: ClassVar[str] = ""
    parameter_power: ClassVar[int] = 1

    def shift_coefficient_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
    ):
        """Return the signed transition-shift coefficient in eV."""
        return (
            -transition.yukawa_difference_ev(mediator_mass_ev)
            / (4 * np.pi)
        )


@dataclass(frozen=True)
class HiggsMixing:
    """Higgs-mixing sensitivity model for kaonic atoms."""

    proton_coupling: float = -1.13e-3
    neutron_coupling: float = -1.14e-3
    kaon_strange_coupling: float = -5 / 3
    kaon_down_coupling: float = 4 / 9

    name: ClassVar[str] = "Higgs mixing"
    parameter_label: ClassVar[str] = r"$\sin\theta$"
    parameter_unit: ClassVar[str] = ""
    parameter_power: ClassVar[int] = 2

    def nuclear_coupling(self, transition: Transition) -> float:
        """Return ``Z g_p + (A - Z) g_n``."""
        atom = transition.atom
        return (
            atom.Z * self.proton_coupling
            + (atom.A - atom.Z) * self.neutron_coupling
        )

    def kaon_coupling(self, transition: Transition, mediator_mass_ev):
        """Return the mass-dependent effective kaon coupling."""
        self._validate_kaonic(transition)
        mediator_mass = np.asarray(mediator_mass_ev, dtype=float)
        kaon_mass_ev = transition.atom.orbiting_mass_ev
        return (
            2
            * (
                self.kaon_strange_coupling
                + self.kaon_down_coupling
            )
            - self.kaon_down_coupling
            * (mediator_mass / kaon_mass_ev) ** 2
        )

    def shift_coefficient_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
    ):
        """Return ``K(m_X)`` in ``delta_E = K sin(vartheta)**2``."""
        self._validate_kaonic(transition)
        return (
            self.nuclear_coupling(transition)
            * self.kaon_coupling(transition, mediator_mass_ev)
            * transition.atom.orbiting_mass_ev
            / (16 * np.pi * HIGGS_VEV_EV)
            * transition.yukawa_difference_ev(mediator_mass_ev)
        )

    @staticmethod
    def _validate_kaonic(transition: Transition) -> None:
        if int(transition.atom.orbiting_particle.pdgid) != -321:
            raise ValueError("HiggsMixing is defined for kaonic atoms")
