"""Custom pylint plugin for ML-specific code smell checks."""

print("pylint_ml_smells plugin __init__ loaded")

from pylint_ml_smells.hyperparameter_checker import HyperparameterNotSetChecker
from pylint_ml_smells.zero_division_checker import ZeroDivisionArgChecker

def register(linter):
    """Register all custom checkers."""
    print("Registering custom ML smell checkers")
    linter.register_checker(HyperparameterNotSetChecker(linter))
    linter.register_checker(ZeroDivisionArgChecker(linter))
