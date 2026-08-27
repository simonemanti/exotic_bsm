"""Nonrelativistic vector-mediator kernels for circular exotic atoms."""

from dataclasses import dataclass

import numpy as np

from .transitions import Transition


@dataclass(frozen=True)
class VectorTransitionCoefficient:
    """Spin-independent and spin-dependent transition coefficients."""

    spin_independent_ev: object
    spin_dependent_ev: object

    @property
    def total_ev(self):
        """Return the sum of the SI and SD coefficients."""
        return self.spin_independent_ev + self.spin_dependent_ev


class VectorInteraction:
    r"""Atomic kernel for a massive vector mediator.

    Couplings use the convention

    .. math::

       V_{\rm SI}(r) = -\frac{g_o^V g_A^V}{4\pi}
       \frac{e^{-m_V r}}{r}.

    The SD term is the leading spin-orbit kernel for a spin-zero orbiting
    particle and a nucleus of spin ``S``.  It is kept separate because an
    unresolved experimental line requires an explicit hyperfine-average
    prescription.
    """

    @staticmethod
    def allowed_total_angular_momenta(
        orbital_angular_momentum: int,
        nuclear_spin: float,
    ) -> np.ndarray:
        """Return allowed ``J`` values from coupling ``L`` and ``S``."""
        VectorInteraction._validate_angular_momenta(
            orbital_angular_momentum,
            nuclear_spin,
        )
        j_min_twice = round(2 * abs(orbital_angular_momentum - nuclear_spin))
        j_max_twice = round(2 * (orbital_angular_momentum + nuclear_spin))
        return np.arange(j_min_twice, j_max_twice + 1, 2) / 2

    @staticmethod
    def spin_independent_level_coefficient_ev(
        transition: Transition,
        n: int,
        mediator_mass_ev,
    ):
        """Return the SI level coefficient per unit ``g_o^V g_A^V``."""
        return -transition.atom.yukawa_element_ev(
            n,
            mediator_mass_ev,
        ) / (4 * np.pi)

    @staticmethod
    def spin_dependent_level_coefficient_ev(
        transition: Transition,
        n: int,
        total_angular_momentum: float,
        nuclear_spin: float,
        mediator_mass_ev,
    ):
        """Return the SD spin-orbit coefficient for one hyperfine level.

        The coefficient multiplies the SD coupling product.  Circular states
        have ``L = n - 1``.  The analytic radial expression is singular for
        ``n = 1``, where this spin-orbit approximation is not applicable.
        """
        transition.atom._validate_n(n)
        if n == 1:
            raise ValueError("the vector spin-orbit kernel requires n >= 2")

        orbital = n - 1
        VectorInteraction._validate_total_angular_momentum(
            orbital,
            nuclear_spin,
            total_angular_momentum,
        )
        mass = np.asarray(mediator_mass_ev, dtype=float)
        if np.any(mass < 0):
            raise ValueError("mediator masses cannot be negative")

        reduced_mass = transition.atom.reduced_mass_ev
        # For a circular Coulomb state: 1/r_n = Z alpha mu / n^2.
        inverse_radius = transition.atom.yukawa_element_ev(n, 0.0)
        radius = 1.0 / inverse_radius
        yukawa_element = transition.atom.yukawa_element_ev(n, mass)

        angular_factor = (
            total_angular_momentum * (total_angular_momentum + 1)
            - orbital * (orbital + 1)
            - nuclear_spin * (nuclear_spin + 1)
        )
        radial_factor = (
            mass / (2 * n + radius * mass)
            + 1.0 / ((2 * n - 2) * radius)
        )
        radial_factor *= (
            4 * n**2
            / (2 * n - 1)
            * inverse_radius
            * (1 + radius * mass / (2 * n)) ** 2
            * yukawa_element
        )
        return angular_factor * radial_factor / (4 * np.pi * reduced_mass)

    def transition_coefficients_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
        nuclear_spin: float,
        j_initial: float,
        j_final: float,
    ) -> VectorTransitionCoefficient:
        """Return signed SI and SD coefficients for one hyperfine line."""
        si = self.spin_independent_level_coefficient_ev(
            transition,
            transition.n_initial,
            mediator_mass_ev,
        ) - self.spin_independent_level_coefficient_ev(
            transition,
            transition.n_final,
            mediator_mass_ev,
        )
        sd = self.spin_dependent_level_coefficient_ev(
            transition,
            transition.n_initial,
            j_initial,
            nuclear_spin,
            mediator_mass_ev,
        ) - self.spin_dependent_level_coefficient_ev(
            transition,
            transition.n_final,
            j_final,
            nuclear_spin,
            mediator_mass_ev,
        )
        return VectorTransitionCoefficient(si, sd)

    def hyperfine_average_coefficients_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
        nuclear_spin: float,
        weighting: str = "degeneracy",
        e1_only: bool = True,
    ) -> VectorTransitionCoefficient:
        """Average unresolved hyperfine components.

        ``weighting="equal"`` assigns the same weight to every component;
        ``weighting="degeneracy"`` uses
        ``(2 J_i + 1)(2 J_f + 1)``.  Neither choice is a calculation of
        electromagnetic transition strengths.
        """
        if weighting not in {"equal", "degeneracy"}:
            raise ValueError("weighting must be 'equal' or 'degeneracy'")
        components = self.hyperfine_components(
            transition,
            nuclear_spin,
            e1_only=e1_only,
        )

        sd_coefficients = []
        weights = []
        si_coefficient = None
        for j_initial, j_final in components:
            coefficients = self.transition_coefficients_ev(
                transition,
                mediator_mass_ev,
                nuclear_spin,
                j_initial,
                j_final,
            )
            si_coefficient = coefficients.spin_independent_ev
            sd_coefficients.append(coefficients.spin_dependent_ev)
            if weighting == "equal":
                weights.append(1.0)
            else:
                weights.append((2 * j_initial + 1) * (2 * j_final + 1))

        averaged_sd = np.average(sd_coefficients, axis=0, weights=weights)
        return VectorTransitionCoefficient(si_coefficient, averaged_sd)

    def hyperfine_components(
        self,
        transition: Transition,
        nuclear_spin: float,
        e1_only: bool = True,
    ) -> list[tuple[float, float]]:
        """Return candidate ``(J_i, J_f)`` hyperfine components.

        With ``e1_only=True``, impose ``Delta J = 0, +/-1`` and exclude
        ``0 -> 0``.  Radial line strengths and cascade populations are not
        included.
        """
        initial_js = self.allowed_total_angular_momenta(
            transition.l_initial,
            nuclear_spin,
        )
        final_js = self.allowed_total_angular_momenta(
            transition.l_final,
            nuclear_spin,
        )
        components = []
        for j_initial in initial_js:
            for j_final in final_js:
                if e1_only and (
                    abs(j_initial - j_final) > 1
                    or (j_initial == 0 and j_final == 0)
                ):
                    continue
                components.append((float(j_initial), float(j_final)))
        return components

    @staticmethod
    def _validate_angular_momenta(
        orbital_angular_momentum: int,
        nuclear_spin: float,
    ) -> None:
        if (
            not isinstance(orbital_angular_momentum, (int, np.integer))
            or orbital_angular_momentum < 0
        ):
            raise ValueError("orbital angular momentum must be a non-negative integer")
        if not np.isfinite(nuclear_spin) or nuclear_spin < 0:
            raise ValueError("nuclear spin must be non-negative and finite")
        if not np.isclose(2 * nuclear_spin, round(2 * nuclear_spin)):
            raise ValueError("nuclear spin must be an integer or half-integer")

    @staticmethod
    def _validate_total_angular_momentum(
        orbital_angular_momentum: int,
        nuclear_spin: float,
        total_angular_momentum: float,
    ) -> None:
        allowed = VectorInteraction.allowed_total_angular_momenta(
            orbital_angular_momentum,
            nuclear_spin,
        )
        if not np.isfinite(total_angular_momentum) or not np.any(
            np.isclose(total_angular_momentum, allowed)
        ):
            raise ValueError(
                f"J={total_angular_momentum} is not allowed for "
                f"L={orbital_angular_momentum}, S={nuclear_spin}"
            )
