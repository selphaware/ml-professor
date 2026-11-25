"""mlprofessor: how predictable is one target column of one table.

This copy is the gate's reference implementation of spec.md. It is a judge
fixture, written for the gate to measure against. It is not the product.
"""
from . import errors, extensions, stages
from .core import ProfileResult, profile

__all__ = ["profile", "ProfileResult", "errors", "extensions", "stages"]
__version__ = "0.1.0"
