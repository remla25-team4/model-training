"""Custom pylint plugin for ML-specific code smell checks."""

from pylint_ml_smells.hyperparameter_checker import HyperparameterNotSetChecker

def register(linter):
    """Register the custom checker with the linter."""
    linter.register_checker(HyperparameterNotSetChecker(linter))
