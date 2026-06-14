"""Core feature model: ``FeatureSpec`` and the ``Feature`` protocol.

A *feature* is not just a flag in the forward pass. It may touch model
parameters, buffers, the forward graph, the optimizer param table, the data
loader, the schedule, warmup shapes, distributed broadcasts, logging metadata
and artifact outputs. Each feature therefore *declares* every surface it
modifies via its :class:`FeatureSpec`, and *applies* its effects to a
``BuildContext`` (see :mod:`nano.builder.context`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nano.builder.context import BuildContext


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative description of everything a feature touches."""

    name: str
    description: str = ""

    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    soft_conflicts: tuple[str, ...] = ()

    owns_params: tuple[str, ...] = ()
    owns_buffers: tuple[str, ...] = ()

    modifies_model: bool = False
    modifies_forward: bool = False
    modifies_optimizer: bool = False
    modifies_schedule: bool = False
    modifies_data: bool = False
    modifies_warmup: bool = False
    modifies_distributed: bool = False
    modifies_loss: bool = False
    modifies_logging: bool = False

    # Whether the codegen template fully guards this feature (model construction,
    # optimizer table *and* hot forward path) so it can be toggled off and still
    # render a valid, behavior-correct training script. Structural features that
    # are kept always-on for the MVP set this to ``False`` -- the builder refuses
    # to disable them rather than emit a broken script.
    template_toggleable: bool = False

    # Allele slot: features sharing an ``allele_group`` are mutually-exclusive
    # variants occupying the same named slot (e.g. the ``orthogonalizer`` slot
    # holds ``polar_express`` | ``newton_schulz``). The builder rejects enabling
    # two members of one group, and rendering requires *exactly one* member of
    # every group to be selected. Allele members are ``template_toggleable`` (the
    # template renders whichever is chosen); the "exactly one" rule replaces the
    # always-on structural guarantee for the slot.
    allele_group: str | None = None


@runtime_checkable
class Feature(Protocol):
    """A feature exposes its :class:`FeatureSpec` and an ``apply`` method."""

    spec: FeatureSpec

    def apply(self, ctx: "BuildContext") -> None:  # pragma: no cover - protocol
        ...


@dataclass
class FunctionFeature:
    """Concrete :class:`Feature` backed by a plain ``apply`` function.

    Lets feature modules declare features as small functions decorated with
    :func:`nano.features.registry.feature` instead of writing a class each time.
    """

    spec: FeatureSpec
    _apply: Callable[["BuildContext"], None]

    def apply(self, ctx: "BuildContext") -> None:
        self._apply(ctx)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Feature({self.spec.name!r})"
