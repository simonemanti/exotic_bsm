"""Plot sensitivity curves."""

from collections.abc import Sequence
from pathlib import Path

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
    title: str | None = None,
    ax=None,
    save_path: str | Path | None = None,
    styles: Sequence[dict] | None = None,
    legend_ncol: int = 1,
    **plot_kwargs,
):
    """Plot one or more sensitivity curves and return ``(figure, axes)``."""
    if isinstance(curves, SensitivityCurve):
        curves = [curves]
    else:
        curves = list(curves)

    if not curves:
        raise ValueError("at least one sensitivity curve is required")
    if styles is not None and len(styles) != len(curves):
        raise ValueError("styles must contain one dictionary per curve")
    if legend_ncol < 1:
        raise ValueError("legend_ncol must be positive")
    if mass_unit not in MASS_SCALES_EV:
        allowed = ", ".join(MASS_SCALES_EV)
        raise ValueError(f"mass_unit must be one of: {allowed}")

    first_curve = curves[0]
    if any(
        curve.parameter_label != first_curve.parameter_label
        or curve.parameter_unit != first_curve.parameter_unit
        for curve in curves[1:]
    ):
        raise ValueError("all curves must use the same parameter and unit")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7.2, 5.0))
    else:
        figure = ax.figure

    mass_scale_ev = MASS_SCALES_EV[mass_unit]
    plotted_masses = []
    for index, curve in enumerate(curves):
        kwargs = dict(plot_kwargs)
        if styles is not None:
            kwargs.update(styles[index])
        kwargs.setdefault("label", curve.transition_label)
        mass_values = curve.mediator_mass_ev / mass_scale_ev
        if np.any(mass_values <= 0):
            raise ValueError("mediator masses must be positive for a log plot")
        plotted_masses.append(mass_values)
        ax.loglog(
            mass_values,
            curve.parameter_limit,
            **kwargs,
        )

    parameter_unit = (
        f" [{first_curve.parameter_unit}]"
        if first_curve.parameter_unit
        else ""
    )
    ax.set_xlabel(rf"$m_X$ [{mass_unit}]")
    ax.set_ylabel(f"{first_curve.parameter_label}{parameter_unit}")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=legend_ncol)
    ax.set_xlim(
        min(np.min(values) for values in plotted_masses),
        max(np.max(values) for values in plotted_masses),
    )
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=200)
    plt.show()
    return figure, ax
