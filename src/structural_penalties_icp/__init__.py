"""
Structural Penalties ICP
========================

A Scikit-Learn-compatible Python package for Inductive Conformal Prediction (ICP)
in multi-label classification tasks using dynamic structural penalties.
"""

from .icp_wrapper import ICPWrapper
from .icp_predictor import InductiveConformalPredictor
from .prediction_regions import PredictionRegions

__version__ = "1.0"

__all__ = [
    "ICPWrapper",
    "InductiveConformalPredictor",
    "PredictionRegions",
]