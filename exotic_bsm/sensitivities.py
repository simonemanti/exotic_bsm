"""Expected sensitivities to interaction-model parameters."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .measurements import Measurement
from .transitions import Transition


class SensitivityModel(Protocol):
    """Interface required by ``expected_sensitivity``."""

    name: str
    parameter_label: str
    parameter_unit: str
    parameter_power: int

    def shift_coefficient_ev(
        self,
        transition: Transition,
        mediator_mass_ev,
    ):
        """Return ``K(m_X)`` in ``delta_E = K * parameter**power``."""


@dataclass(frozen=True)
class SensitivityCurve:
    """Expected parameter limit as a function of mediator mass."""

    mediator_mass_ev: np.ndarray
    parameter_limit: np.ndarray
    model_name: str
    parameter_label: str
    parameter_unit: str
    transition_label: str
    confidence_multiplier: float


def expected_sensitivity(
    measurement: Measurement,
    model: SensitivityModel,
    mediator_mass_ev,
    confidence_multiplier: float = 2.0,
) -> SensitivityCurve:
    """Calculate an uncertainty-only expected sensitivity curve.

    For ``delta_E = K(m_X) * parameter**p``, the returned limit is

        [confidence_multiplier * sigma_E / abs(K(m_X))] ** (1 / p).
    """
    masses = np.asarray(mediator_mass_ev, dtype=float)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("mediator_mass_ev must be a non-empty 1D array")
    if not np.all(np.isfinite(masses)):
        raise ValueError("mediator masses must be finite")
    if np.any(masses < 0):
        raise ValueError("mediator masses cannot be negative")
    if not np.isfinite(confidence_multiplier) or confidence_multiplier <= 0:
        raise ValueError("confidence_multiplier must be positive and finite")

    parameter_power = model.parameter_power
    if (
        not isinstance(parameter_power, int)
        or isinstance(parameter_power, bool)
        or parameter_power <= 0
    ):
        raise ValueError("model.parameter_power must be a positive integer")

    coefficient_ev = np.asarray(
        model.shift_coefficient_ev(
            measurement.transition,
            masses,
        ),
        dtype=float,
    )
    if coefficient_ev.shape != masses.shape:
        raise ValueError("model coefficient must match the mediator-mass shape")
    if np.any(np.isnan(coefficient_ev)):
        raise ValueError("model coefficient cannot contain NaN")

    threshold_ev = (
        confidence_multiplier
        * measurement.total_uncertainty_ev
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        parameter_limit = (
            threshold_ev / np.abs(coefficient_ev)
        ) ** (1 / parameter_power)

    return SensitivityCurve(
        mediator_mass_ev=np.array(masses, copy=True),
        parameter_limit=np.array(parameter_limit, copy=True),
        model_name=model.name,
        parameter_label=model.parameter_label,
        parameter_unit=model.parameter_unit,
        transition_label=measurement.transition.display_label,
        confidence_multiplier=float(confidence_multiplier),
    )
