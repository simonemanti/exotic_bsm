"""Plot sensitivity curves."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .sensitivities import SensitivityCurve


MASS_SCALES_EV = {
    "eV": 1.0,
    "keV": 1e3,
    "MeV": 1e6,
    "GeV": 1e9,
}


def plot_sensitivity(
    curves: SensitivityCurve | Sequence[SensitivityCurve],
    *,
    mass_unit: str = "eV",
    ax=None,
    styles: Sequence[dict] | None = None,
    **plot_kwargs,
):
    """Draw one or more sensitivity curves and return ``(figure, axes)``."""
    if isinstance(curves, SensitivityCurve):
        curves = [curves]
    else:
        curves = list(curves)

    if not curves:
        raise ValueError("at least one sensitivity curve is required")
    if styles is not None and len(styles) != len(curves):
        raise ValueError("styles must contain one dictionary per curve")
    if mass_unit not in MASS_SCALES_EV:
        allowed = ", ".join(MASS_SCALES_EV)
        raise ValueError(f"mass_unit must be one of: {allowed}")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7.2, 5.0))
    else:
        figure = ax.figure

    mass_scale_ev = MASS_SCALES_EV[mass_unit]
    for index, curve in enumerate(curves):
        kwargs = dict(plot_kwargs)
        if styles is not None:
            kwargs.update(styles[index])
        kwargs.setdefault("label", curve.transition_label)
        mass_values = curve.mediator_mass_ev / mass_scale_ev
        if np.any(mass_values <= 0):
            raise ValueError("mediator masses must be positive for a log plot")
        ax.loglog(
            mass_values,
            curve.parameter_limit,
            **kwargs,
        )
    return figure, ax
