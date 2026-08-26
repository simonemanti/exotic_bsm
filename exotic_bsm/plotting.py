"""Plot sensitivity curves."""

from collections.abc import Sequence

import matplotlib.pyplot as plt

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
    **plot_kwargs,
):
    """Plot one or more sensitivity curves and return ``(figure, axes)``."""
    if isinstance(curves, SensitivityCurve):
        curves = [curves]
    else:
        curves = list(curves)

    if not curves:
        raise ValueError("at least one sensitivity curve is required")
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
    for curve in curves:
        kwargs = dict(plot_kwargs)
        kwargs.setdefault("label", curve.transition_label)
        ax.loglog(
            curve.mediator_mass_ev / mass_scale_ev,
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
    ax.legend()
    figure.tight_layout()
    plt.show()
    return figure, ax
