"""Interaction models for exotic-atom energy shifts."""

from typing import ClassVar

import numpy as np

from .transitions import Transition


class YukawaCoupling:
    """Generic Yukawa interaction parameterized by ``g_o * g_A``.

    The potential convention is

        V(r) = -(g_o * g_A) exp(-m_X r) / (4 pi r).

    Consequently, ``shift_coefficient_ev`` returns the coefficient ``K`` in
    ``delta_E = K * (g_o * g_A)``.
    """

    name: ClassVar[str] = "Yukawa coupling"
    parameter_label: ClassVar[str] = r"$|g_o g_A|$"
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
