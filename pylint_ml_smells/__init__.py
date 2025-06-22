from pylint_ml_smells.hyperparameter_checker import HyperparameterNotSetChecker

def register(linter):
    linter.register_checker(HyperparameterNotSetChecker(linter))
