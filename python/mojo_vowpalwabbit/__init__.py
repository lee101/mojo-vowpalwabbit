"""Online linear learning with Vowpal Wabbit-compatible hashing and API."""

from . import pyvw
from .pyvw import (
    Example,
    ExampleBatch,
    ExampleNamespace,
    LabelType,
    NamespaceId,
    PredictionType,
    SimpleLabel,
    Workspace,
    vw,
)

__version__ = "0.1.0"
__all__ = [
    "Example",
    "ExampleBatch",
    "ExampleNamespace",
    "LabelType",
    "NamespaceId",
    "PredictionType",
    "SimpleLabel",
    "Workspace",
    "pyvw",
    "vw",
]
