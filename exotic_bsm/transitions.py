"""Transitions between circular levels of exotic atoms."""

from dataclasses import dataclass
from numbers import Integral

from .atoms import ExoticAtom


@dataclass(frozen=True)
class Transition:
    """Transition between two circular states of an exotic atom."""

    atom: ExoticAtom
    n_initial: int
    n_final: int
    label: str | None = None

    def __post_init__(self) -> None:
        self._validate_n(self.n_initial, "n_initial")
        self._validate_n(self.n_final, "n_final")
        if self.n_initial <= self.n_final:
            raise ValueError("n_initial must be larger than n_final")
        if self.label is not None and not self.label.strip():
            raise ValueError("label cannot be empty")

    @property
    def l_initial(self) -> int:
        """Orbital angular momentum of the initial circular state."""
        return self.n_initial - 1

    @property
    def l_final(self) -> int:
        """Orbital angular momentum of the final circular state."""
        return self.n_final - 1

    @property
    def display_label(self) -> str:
        """User label, or a compact automatically generated label."""
        if self.label is not None:
            return self.label
        return (
            f"{self.atom.orbiting_name} "
            f"{self.atom.nucleus_particle.name} "
            f"{self.n_initial}->{self.n_final}"
        )

    @property
    def coulomb_energy_ev(self) -> float:
        """Positive leading-order Coulomb transition energy in eV."""
        return abs(
            self.atom.coulomb_energy_ev(self.n_initial)
            - self.atom.coulomb_energy_ev(self.n_final)
        )

    def yukawa_difference_ev(self, mediator_mass_ev):
        """Signed difference of the initial and final Yukawa elements."""
        return self.atom.yukawa_element_ev(
            self.n_initial,
            mediator_mass_ev,
        ) - self.atom.yukawa_element_ev(
            self.n_final,
            mediator_mass_ev,
        )

    @staticmethod
    def _validate_n(value: int, name: str) -> None:
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
