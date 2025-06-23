"""Custom Pylint checker for ML model hyperparameter code smells."""

from pylint.checkers import BaseChecker
import astroid


class HyperparameterNotSetChecker(BaseChecker):
    """Checker that warns when ML models are instantiated without explicit hyperparameters."""
    name = "hyperparameter-checker"
    priority = -1
    msgs = {
        "C9001": (
	    "Hyperparameter not explicitly set in ML model instantiation",
	    "hyperparameter-not-set",
	    "Set at least one hyperparameter explicitly when instantiating a "
	    "machine learning model.",
	),
    }

    TARGET_CLASSES = { "GaussianNB" }

    def visit_call(self, node):
        """Visit each function and check for ML model instantiations without hyperparameters."""
        try:
            if isinstance(node.func, astroid.Name):
                func_name = node.func.name
            elif isinstance(node.func, astroid.Attribute):
                func_name = node.func.attrname
            else:
                return

            if func_name in self.TARGET_CLASSES:
                has_kwargs = any(node.keywords)
                if not has_kwargs:
                    self.add_message("hyperparameter-not-set", node=node)
        except (AttributeError, KeyError) as e:
            print(f"Checker failed on node: {e}")
