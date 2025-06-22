from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH
import astroid


class HyperparameterNotSetChecker(BaseChecker):

    name = "hyperparameter-checker"
    priority = -1
    msgs = {
        "C9001": (
            "Hyperparameter not explicitly set in ML model instantiation",
            "hyperparameter-not-set",
            "Set at least one hyperparameter explicitly when instantiating a machine learning model.",
        ),
    }

    # Target specific ML classes
    TARGET_CLASSES = { "GaussianNB" }

    def visit_call(self, node):
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

        except Exception:
            pass  # to not crash Pylint
