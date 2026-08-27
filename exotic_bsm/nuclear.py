"""Curated local nuclear-spin data for spin-dependent BSM models.

The stored quantities are zero-momentum nuclear expectation values
``<S_p>`` and ``<S_n>``. They are signed model-dependent matrix elements,
not normalized fractions, and therefore need neither lie in ``[0, 1]`` nor
sum to one.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


SPIN_DATA_REVIEW_URL = "https://arxiv.org/abs/hep-ph/0406218"
NUBASE2020_URL = "https://doi.org/10.1088/1674-1137/abddae"
BORON10_EVALUATION_URL = (
    "https://nucldata.tunl.duke.edu/nucldata/HTML/A=10/10B_2004.shtml"
)


@dataclass(frozen=True)
class NuclearStructure:
    """Ground-state spin and proton/neutron spin expectation values."""

    Z: int
    A: int
    spin: float
    parity: int
    proton_spin_expectation: float
    neutron_spin_expectation: float
    spin_source: str
    spin_expectation_source: str
    spin_expectation_method: str

    def __post_init__(self) -> None:
        if self.Z <= 0:
            raise ValueError("Z must be positive")
        if self.A < self.Z:
            raise ValueError("A must be greater than or equal to Z")
        if not np.isfinite(self.spin) or self.spin < 0:
            raise ValueError("spin must be non-negative and finite")
        if not np.isclose(2 * self.spin, round(2 * self.spin)):
            raise ValueError("spin must be an integer or half-integer")
        if self.parity not in (-1, 1):
            raise ValueError("parity must be +1 or -1")

        expectations = (
            self.proton_spin_expectation,
            self.neutron_spin_expectation,
        )
        if not all(np.isfinite(value) for value in expectations):
            raise ValueError("spin expectation values must be finite")
        if self.spin == 0 and not all(
            np.isclose(value, 0.0) for value in expectations
        ):
            raise ValueError("spin-zero nuclei must have zero spin expectations")
        if not self.spin_source.strip():
            raise ValueError("spin_source cannot be empty")
        if not self.spin_expectation_source.strip():
            raise ValueError("spin_expectation_source cannot be empty")
        if not self.spin_expectation_method.strip():
            raise ValueError("spin_expectation_method cannot be empty")

    @property
    def spin_parity(self) -> str:
        """Return a compact spectroscopic label such as ``1/2+``."""
        doubled_spin = round(2 * self.spin)
        if doubled_spin % 2:
            spin_label = f"{doubled_spin}/2"
        else:
            spin_label = str(doubled_spin // 2)
        parity_label = "+" if self.parity > 0 else "-"
        return f"{spin_label}{parity_label}"

    @property
    def proton_spin_fraction(self) -> float:
        """Deprecated compatibility alias for ``proton_spin_expectation``."""
        return self.proton_spin_expectation

    @property
    def neutron_spin_fraction(self) -> float:
        """Deprecated compatibility alias for ``neutron_spin_expectation``."""
        return self.neutron_spin_expectation


_SPIN_SOURCE = f"NUBASE2020 adopted ground-state J^pi; {NUBASE2020_URL}"
_REVIEW_SOURCE = f"Bednyakov and Simkovic (2005), Tables I-II; {SPIN_DATA_REVIEW_URL}"

_NUCLEAR_STRUCTURE_TABLE = {
    (2, 3): NuclearStructure(
        2, 3, 0.5, +1, -0.021, 0.462, _SPIN_SOURCE, _REVIEW_SOURCE,
        "extended odd-group model, g_A/g_V = 1.25",
    ),
    (5, 10): NuclearStructure(
        5, 10, 3.0, +1, 0.5, 0.5,
        f"TUNL A=10 evaluation; {BORON10_EVALUATION_URL}",
        f"estimate from the evaluated 3+ ground state; {BORON10_EVALUATION_URL}",
        "minimal isoscalar LS-coupling estimate with equal proton/neutron sharing",
    ),
    (5, 11): NuclearStructure(
        5, 11, 1.5, -1, 0.292, 0.008, _SPIN_SOURCE, _REVIEW_SOURCE,
        "Pacheco-Strottman shell model",
    ),
    (6, 12): NuclearStructure(
        6, 12, 0.0, +1, 0.0, 0.0, _SPIN_SOURCE,
        "exact consequence of J=0", "angular-momentum selection rule",
    ),
    (9, 19): NuclearStructure(
        9, 19, 0.5, +1, 0.4751, -0.0087, _SPIN_SOURCE, _REVIEW_SOURCE,
        "Divari et al. shell model",
    ),
    (10, 20): NuclearStructure(
        10, 20, 0.0, +1, 0.0, 0.0, _SPIN_SOURCE,
        "exact consequence of J=0", "angular-momentum selection rule",
    ),
}

NUCLEAR_STRUCTURE_TABLE = MappingProxyType(_NUCLEAR_STRUCTURE_TABLE)


def get_nuclear_structure(Z: int, A: int) -> NuclearStructure:
    """Return curated local nuclear data for isotope ``(Z, A)``."""
    try:
        return NUCLEAR_STRUCTURE_TABLE[(Z, A)]
    except KeyError as error:
        raise KeyError(
            f"no local nuclear-structure data for Z={Z}, A={A}"
        ) from error
