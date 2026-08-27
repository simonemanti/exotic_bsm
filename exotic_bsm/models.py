"""Interaction models for exotic-atom energy shifts."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .constants import ALPHA, HIGGS_VEV_EV
from .nuclear import get_nuclear_structure
from .transitions import Transition
from .vector import VectorInteraction


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
class DarkPhoton:
    r"""Dark-photon sensitivity model for exotic atoms.

    ``mode`` may be ``"spin_independent"``, ``"hyperfine"``, or
    ``"average"``.  A hyperfine component requires explicit ``j_initial``
    and ``j_final``.  The average mode uses the schematic degeneracy average
    implemented by :class:`~exotic_bsm.vector.VectorInteraction`.

    The canonical kinetic-mixing potential is obtained with
    ``yukawa_normalization=1``.  A different multiplicative normalization
    can be requested explicitly when reproducing an external convention.
    """

    mode: str = "spin_independent"
    j_initial: float | None = None
    j_final: float | None = None
    yukawa_normalization: float = 1.0
    average_weighting: str = "degeneracy"
    e1_only: bool = True

    name: ClassVar[str] = "Dark photon"
    parameter_label: ClassVar[str] = r"$\varepsilon$"
    parameter_unit: ClassVar[str] = ""
    parameter_power: ClassVar[int] = 2

    def __post_init__(self) -> None:
        allowed_modes = {"spin_independent", "hyperfine", "average"}
        if self.mode not in allowed_modes:
            raise ValueError(
                f"mode must be one of {sorted(allowed_modes)}"
            )
        if not np.isfinite(self.yukawa_normalization):
            raise ValueError("yukawa_normalization must be finite")
        if self.yukawa_normalization <= 0:
            raise ValueError("yukawa_normalization must be positive")

        has_initial = self.j_initial is not None
        has_final = self.j_final is not None
        if self.mode == "hyperfine" and not (has_initial and has_final):
            raise ValueError(
                "hyperfine mode requires j_initial and j_final"
            )
        if self.mode != "hyperfine" and (has_initial or has_final):
            raise ValueError(
                "j_initial and j_final are only valid in hyperfine mode"
            )
        if self.average_weighting not in {"equal", "degeneracy"}:
            raise ValueError(
                "average_weighting must be 'equal' or 'degeneracy'"
            )

    @staticmethod
    def electric_coupling_product_per_epsilon_squared(
        transition: Transition,
    ) -> float:
        """Return ``g_K g_A / epsilon**2 = -4 pi alpha Z``."""
        _validate_kaonic(transition)
        return -4 * np.pi * ALPHA * transition.atom.Z

    @staticmethod
    def spin_coupling_product_per_epsilon_squared(
        transition: Transition,
    ) -> float:
        """Return the effective proton-spin coupling per ``epsilon**2``.

        The nuclear factor is the fraction ``<S_p>/J`` from the curated
        local nuclear table.  This is a leading effective prescription; the
        magnetic-current normalization must be validated before precision
        use of the SD contribution.
        """
        _validate_kaonic(transition)
        structure = get_nuclear_structure(
            transition.atom.Z,
            transition.atom.A,
        )
        if structure.spin == 0:
            return 0.0
        proton_spin_fraction = (
            structure.proton_spin_expectation / structure.spin
        )
        return 4 * np.pi * ALPHA * proton_spin_fraction

    @staticmethod
    def nuclear_spin(transition: Transition) -> float:
        """Return the ground-state nuclear spin from the local table."""
        return get_nuclear_structure(
            transition.atom.Z,
            transition.atom.A,
        ).spin

    def hyperfine_components(
        self,
        transition: Transition,
    ) -> list[tuple[float, float]]:
        """Return candidate hyperfine components for the transition."""
        return VectorInteraction().hyperfine_components(
            transition,
            self.nuclear_spin(transition),
            e1_only=self.e1_only,
        )

    def shift_coefficient_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
    ):
        """Return ``K`` in ``delta_E = K * epsilon**2``."""
        _validate_kaonic(transition)
        kernel = VectorInteraction()
        electric_coupling = (
            self.yukawa_normalization
            * self.electric_coupling_product_per_epsilon_squared(
                transition
            )
        )

        if self.mode == "spin_independent":
            initial_coefficient = (
                kernel.spin_independent_level_coefficient_ev(
                    transition,
                    transition.n_initial,
                    mediator_mass_ev,
                )
            )
            final_coefficient = (
                kernel.spin_independent_level_coefficient_ev(
                    transition,
                    transition.n_final,
                    mediator_mass_ev,
                )
            )
            return electric_coupling * (
                initial_coefficient - final_coefficient
            )

        nuclear_spin = self.nuclear_spin(transition)
        if self.mode == "hyperfine":
            coefficient = kernel.transition_coefficients_ev(
                transition,
                mediator_mass_ev,
                nuclear_spin=nuclear_spin,
                j_initial=self.j_initial,
                j_final=self.j_final,
            )
        else:
            coefficient = kernel.hyperfine_average_coefficients_ev(
                transition,
                mediator_mass_ev,
                nuclear_spin=nuclear_spin,
                weighting=self.average_weighting,
                e1_only=self.e1_only,
            )

        spin_coupling = self.spin_coupling_product_per_epsilon_squared(
            transition
        )
        return (
            electric_coupling * coefficient.spin_independent_ev
            + spin_coupling * coefficient.spin_dependent_ev
        )


@dataclass(frozen=True)
class ProtophobicVector:
    r"""Protophobic vector model with dominant neutron coupling.

    The available modes and averaging conventions are the same as for
    :class:`DarkPhoton`.  ``yukawa_normalization=1`` follows the canonical
    ``1/(4 pi)`` potential convention; external conventions can request a
    different factor explicitly.
    """

    mode: str = "spin_independent"
    j_initial: float | None = None
    j_final: float | None = None
    yukawa_normalization: float = 1.0
    average_weighting: str = "degeneracy"
    e1_only: bool = True

    name: ClassVar[str] = "Protophobic vector"
    parameter_label: ClassVar[str] = r"$\varepsilon_B$"
    parameter_unit: ClassVar[str] = ""
    parameter_power: ClassVar[int] = 2

    def __post_init__(self) -> None:
        allowed_modes = {"spin_independent", "hyperfine", "average"}
        if self.mode not in allowed_modes:
            raise ValueError(
                f"mode must be one of {sorted(allowed_modes)}"
            )
        if (
            not np.isfinite(self.yukawa_normalization)
            or self.yukawa_normalization <= 0
        ):
            raise ValueError("yukawa_normalization must be positive and finite")
        has_initial = self.j_initial is not None
        has_final = self.j_final is not None
        if self.mode == "hyperfine" and not (has_initial and has_final):
            raise ValueError(
                "hyperfine mode requires j_initial and j_final"
            )
        if self.mode != "hyperfine" and (has_initial or has_final):
            raise ValueError(
                "j_initial and j_final are only valid in hyperfine mode"
            )
        if self.average_weighting not in {"equal", "degeneracy"}:
            raise ValueError(
                "average_weighting must be 'equal' or 'degeneracy'"
            )

    @staticmethod
    def vector_coupling_product_per_epsilon_squared(
        transition: Transition,
    ) -> float:
        """Return the coherent coupling ``4 pi alpha (Z-A)``."""
        _validate_kaonic(transition)
        return 4 * np.pi * ALPHA * (
            transition.atom.Z - transition.atom.A
        )

    @staticmethod
    def spin_coupling_product_per_epsilon_squared(
        transition: Transition,
    ) -> float:
        """Return the effective neutron-spin coupling per ``epsilon_B**2``."""
        _validate_kaonic(transition)
        structure = get_nuclear_structure(
            transition.atom.Z,
            transition.atom.A,
        )
        if structure.spin == 0:
            return 0.0
        neutron_spin_fraction = (
            structure.neutron_spin_expectation / structure.spin
        )
        return -4 * np.pi * ALPHA * neutron_spin_fraction

    @staticmethod
    def nuclear_spin(transition: Transition) -> float:
        """Return the ground-state nuclear spin from the local table."""
        return get_nuclear_structure(
            transition.atom.Z,
            transition.atom.A,
        ).spin

    def hyperfine_components(
        self,
        transition: Transition,
    ) -> list[tuple[float, float]]:
        """Return candidate hyperfine components for the transition."""
        return VectorInteraction().hyperfine_components(
            transition,
            self.nuclear_spin(transition),
            e1_only=self.e1_only,
        )

    def shift_coefficient_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
    ):
        """Return ``K`` in ``delta_E = K * epsilon_B**2``."""
        _validate_kaonic(transition)
        kernel = VectorInteraction()
        coherent_coupling = (
            self.yukawa_normalization
            * self.vector_coupling_product_per_epsilon_squared(
                transition
            )
        )

        initial_si = kernel.spin_independent_level_coefficient_ev(
            transition,
            transition.n_initial,
            mediator_mass_ev,
        )
        final_si = kernel.spin_independent_level_coefficient_ev(
            transition,
            transition.n_final,
            mediator_mass_ev,
        )
        transition_si = initial_si - final_si
        if self.mode == "spin_independent":
            return coherent_coupling * transition_si

        nuclear_spin = self.nuclear_spin(transition)
        if self.mode == "hyperfine":
            coefficient = kernel.transition_coefficients_ev(
                transition,
                mediator_mass_ev,
                nuclear_spin=nuclear_spin,
                j_initial=self.j_initial,
                j_final=self.j_final,
            )
        else:
            coefficient = kernel.hyperfine_average_coefficients_ev(
                transition,
                mediator_mass_ev,
                nuclear_spin=nuclear_spin,
                weighting=self.average_weighting,
                e1_only=self.e1_only,
            )

        spin_coupling = self.spin_coupling_product_per_epsilon_squared(
            transition
        )
        return (
            coherent_coupling * coefficient.spin_independent_ev
            + spin_coupling * coefficient.spin_dependent_ev
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
