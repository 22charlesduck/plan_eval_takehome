"""Data inspection command."""


class InspectCommand:
    """Inspect data files and show summary statistics."""

    help_text = "Inspect data and show summary"

    def __init__(self, config, formatter):
        self.config = config
        self.formatter = formatter

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("input", help="Input file to inspect")
        parser.add_argument("--detailed", action="store_true")

    def execute(self, args):
        """Run inspection."""
        raise NotImplementedError("Inspect command not yet implemented")
