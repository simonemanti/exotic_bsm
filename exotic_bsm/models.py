"""Interaction models for exotic-atom energy shifts."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .constants import HIGGS_VEV_EV
from .transitions import Transition


def _validate_kaonic(transition: Transition) -> None:
    if int(transition.atom.orbiting_particle.pdgid) != -321:
        raise ValueError("model is defined for kaonic atoms")


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
        _validate_kaonic(transition)
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
        _validate_kaonic(transition)
        return (
            self.nuclear_coupling(transition)
            * self.kaon_coupling(transition, mediator_mass_ev)
            * transition.atom.orbiting_mass_ev
            / (16 * np.pi * HIGGS_VEV_EV)
            * transition.yukawa_difference_ev(mediator_mass_ev)
        )


@dataclass(frozen=True)
class UDSScalar:
    """Scalar model with equal up, down, and strange couplings."""

    proton_coupling_gev: float = 90.3e-3
    neutron_coupling_gev: float = 92.3e-3
    kaon_strange_coupling_gev: float = -246.0
    kaon_down_coupling_gev: float = 0.0

    name: ClassVar[str] = "UDS scalar"
    parameter_label: ClassVar[str] = r"$1/f_\phi$"
    parameter_unit: ClassVar[str] = r"$\mathrm{GeV}^{-1}$"
    parameter_power: ClassVar[int] = 2

    def nuclear_coupling_gev(self, transition: Transition) -> float:
        """Return ``Z g_p + (A - Z) g_n`` in GeV."""
        atom = transition.atom
        return (
            atom.Z * self.proton_coupling_gev
            + (atom.A - atom.Z) * self.neutron_coupling_gev
        )

    def kaon_coupling_gev(self, transition: Transition, mediator_mass_ev):
        """Return the effective kaon coupling in GeV."""
        _validate_kaonic(transition)
        mediator_mass = np.asarray(mediator_mass_ev, dtype=float)
        kaon_mass_ev = transition.atom.orbiting_mass_ev
        return (
            2
            * (
                self.kaon_strange_coupling_gev
                + self.kaon_down_coupling_gev
            )
            - self.kaon_down_coupling_gev
            * (mediator_mass / kaon_mass_ev) ** 2
        )

    def shift_coefficient_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
    ):
        """Return ``K`` in ``delta_E = K * (1/f_phi)**2``."""
        _validate_kaonic(transition)
        return (
            self.nuclear_coupling_gev(transition)
            * self.kaon_coupling_gev(transition, mediator_mass_ev)
            * transition.atom.orbiting_mass_ev
            / (16 * np.pi * HIGGS_VEV_EV)
            * transition.yukawa_difference_ev(mediator_mass_ev)
        )


@dataclass(frozen=True)
class ALPGlounCPViolating:
    """CP-violating scalar-gluon ALP model for kaonic atoms.

    The unknown parameter is ``C_g / Lambda`` in TeV^-1.  The effective
    The effective hadronic coefficients implement the scalar gluon trace-
    anomaly matching convention used in Eqs. (5.37)-(5.38) of the
    accompanying thesis, with all other ALP operators set to zero.
    """

    proton_mass_gev: float = 0.938272
    neutron_mass_gev: float = 0.939565
    sigma_up_gev: float = 0.017
    sigma_down_proton_gev: float = 0.032
    sigma_strange_proton_gev: float = 0.0413
    sigma_up_neutron_gev: float = 0.015
    sigma_down_neutron_gev: float = 0.036
    sigma_strange_neutron_gev: float = 0.0413

    name: ClassVar[str] = "ALP scalar-gluon CP violating"
    parameter_label: ClassVar[str] = r"$C_g/\Lambda$"
    parameter_unit: ClassVar[str] = r"$\mathrm{TeV}^{-1}$"
    parameter_power: ClassVar[int] = 2

    @staticmethod
    def active_flavors(mediator_mass_ev):
        """Return a stepwise number of active quark flavors."""
        mass = np.asarray(mediator_mass_ev, dtype=float)
        flavors = np.zeros_like(mass)
        flavors[mass >= 2.2e6] = 1
        flavors[mass >= 4.7e6] = 2
        flavors[mass >= 95e6] = 3
        flavors[mass >= 1.28e9] = 4
        flavors[mass >= 4.18e9] = 5
        flavors[mass >= 173e9] = 6
        return flavors

    def kappa_per_cg(self, mediator_mass_ev):
        """Return ``kappa / C_g = 32 pi^2 / beta_QCD``."""
        beta_qcd = 11.0 - (2.0 / 3.0) * self.active_flavors(
            mediator_mass_ev
        )
        return 32.0 * np.pi**2 / beta_qcd

    def proton_coupling_gev(self) -> float:
        """Return the scalar proton coefficient per unit ``C_g`` in GeV."""
        sigma_sum = (
            self.sigma_up_gev
            + self.sigma_down_proton_gev
            + self.sigma_strange_proton_gev
        )
        return (
            -32.0
            * np.pi**2
            / 9.0
            * (self.proton_mass_gev - sigma_sum)
        )

    def neutron_coupling_gev(self) -> float:
        """Return the scalar neutron coefficient per unit ``C_g`` in GeV."""
        sigma_sum = (
            self.sigma_up_neutron_gev
            + self.sigma_down_neutron_gev
            + self.sigma_strange_neutron_gev
        )
        return (
            -32.0
            * np.pi**2
            / 9.0
            * (self.neutron_mass_gev - sigma_sum)
        )

    def nuclear_coupling_tev(self, transition: Transition) -> float:
        """Return ``Z g_p + (A-Z) g_n`` per unit ``C_g`` in TeV."""
        atom = transition.atom
        coupling_gev = (
            atom.Z * self.proton_coupling_gev()
            + (atom.A - atom.Z) * self.neutron_coupling_gev()
        )
        return coupling_gev / 1e3

    def kaon_coupling(self, transition: Transition, mediator_mass_ev):
        """Return the effective kaon coefficient per unit ``C_g``."""
        _validate_kaonic(transition)
        mediator_mass = np.asarray(mediator_mass_ev, dtype=float)
        kappa = self.kappa_per_cg(mediator_mass)
        g_kaon_scalar = -3.0 * kappa
        g_kaon_derivative = 2.0 * kappa
        return (
            2.0 * (g_kaon_scalar + g_kaon_derivative)
            - g_kaon_derivative
            * (mediator_mass / transition.atom.orbiting_mass_ev) ** 2
        )

    def shift_coefficient_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
    ):
        """Return ``K`` in ``delta_E = K * (C_g/Lambda [TeV^-1])**2``."""
        _validate_kaonic(transition)
        kaon_mass_tev = transition.atom.orbiting_mass_ev / 1e12
        return (
            self.nuclear_coupling_tev(transition)
            * self.kaon_coupling(transition, mediator_mass_ev)
            * kaon_mass_tev
            / (16.0 * np.pi)
            * transition.yukawa_difference_ev(mediator_mass_ev)
        )


@dataclass(frozen=True)
class ALPUniversalQuarkCoupling:
    """CP-violating ALP with universal scalar light-quark coupling.

    The model assumes ``y_u = y_d = y_s = y`` and returns sensitivities
    to ``y / Lambda`` expressed in GeV^-1.
    """

    higgs_vev_gev: float = 246.0
    up_quark_mass_gev: float = 2.2e-3
    down_quark_mass_gev: float = 4.7e-3
    strange_quark_mass_gev: float = 95e-3
    sigma_up_proton_gev: float = 17e-3
    sigma_down_proton_gev: float = 32e-3
    sigma_strange_proton_gev: float = 41.3e-3
    sigma_up_neutron_gev: float = 15e-3
    sigma_down_neutron_gev: float = 36e-3
    sigma_strange_neutron_gev: float = 41.3e-3

    name: ClassVar[str] = "ALP universal quark coupling"
    parameter_label: ClassVar[str] = r"$y/\Lambda$"
    parameter_unit: ClassVar[str] = r"$\mathrm{GeV}^{-1}$"
    parameter_power: ClassVar[int] = 2

    def proton_coupling_gev(self) -> float:
        """Return the scalar proton coefficient per unit ``y`` in GeV."""
        return self.higgs_vev_gev * (
            self.sigma_up_proton_gev / self.up_quark_mass_gev
            + self.sigma_down_proton_gev / self.down_quark_mass_gev
            + self.sigma_strange_proton_gev
            / self.strange_quark_mass_gev
        )

    def neutron_coupling_gev(self) -> float:
        """Return the scalar neutron coefficient per unit ``y`` in GeV."""
        return self.higgs_vev_gev * (
            self.sigma_up_neutron_gev / self.up_quark_mass_gev
            + self.sigma_down_neutron_gev / self.down_quark_mass_gev
            + self.sigma_strange_neutron_gev
            / self.strange_quark_mass_gev
        )

    def nuclear_coupling_gev(self, transition: Transition) -> float:
        """Return ``Z g_p + (A-Z) g_n`` per unit ``y`` in GeV."""
        atom = transition.atom
        return (
            atom.Z * self.proton_coupling_gev()
            + (atom.A - atom.Z) * self.neutron_coupling_gev()
        )

    def kaon_coupling(self, transition: Transition, mediator_mass_ev):
        """Return the effective kaon coefficient per unit ``y``."""
        _validate_kaonic(transition)
        mediator_mass = np.asarray(mediator_mass_ev, dtype=float)
        g_kaon_scalar = (
            2.0
            * self.higgs_vev_gev
            / (self.up_quark_mass_gev + self.strange_quark_mass_gev)
        )
        return np.full_like(mediator_mass, 2.0 * g_kaon_scalar)

    def shift_coefficient_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
    ):
        """Return ``K`` in ``delta_E = K * (y/Lambda [GeV^-1])**2``."""
        _validate_kaonic(transition)
        kaon_mass_gev = transition.atom.orbiting_mass_ev / 1e9
        return (
            self.nuclear_coupling_gev(transition)
            * self.kaon_coupling(transition, mediator_mass_ev)
            * kaon_mass_gev
            / (16.0 * np.pi)
            * transition.yukawa_difference_ev(mediator_mass_ev)
        )
