"""Data validation command."""

from cli.core.validators import SchemaValidator, RuleValidator


class ValidateCommand:
    """Validate input data against schema and rules."""

    help_text = "Validate data against schema and rules"

    def __init__(self, config, formatter):
        self.config = config
        self.formatter = formatter

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("input", help="Input file to validate")
        parser.add_argument("--schema", help="Schema file path")
        parser.add_argument("--strict", action="store_true")

    def execute(self, args):
        """Run validation."""
        raise NotImplementedError("Validation command not yet implemented")
