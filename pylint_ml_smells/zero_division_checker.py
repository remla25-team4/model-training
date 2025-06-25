

import astroid
from pylint.checkers import BaseChecker

class ZeroDivisionArgChecker(BaseChecker):
    """Warn when classification_report is called without zero_division kwarg."""

    name = "zero-division-checker"
    priority = -1
    msgs = {
        "W9003": (
            "classification_report missing zero_division argument; pass zero_division=0 or 1",
            "missing-zero-division",
            "Without zero_division set, classification_report may error or warn on empty classes.",
        ),
    }

    def visit_call(self, node):
    	"""Inspected every function call; flag classification_report without zero_division."""
    	if isinstance(node.func, astroid.Name) and node.func.name == "classification_report":           
    		if not any(kw.arg == "zero_division" for kw in node.keywords):
    			self.add_message("missing-zero-division", node=node)

def register(linter):
    linter.register_checker(ZeroDivisionArgChecker(linter))

