"""Main data processing command."""

from cli.core.pipeline import Pipeline
from cli.utils.file_io import read_input, write_output


class ProcessCommand:
    """Execute the data processing pipeline on input files."""

    help_text = "Process input data through the pipeline"

    def __init__(self, config, formatter):
        self.config = config
        self.formatter = formatter
        self.pipeline = Pipeline(config)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("input", help="Input file path")
        parser.add_argument("-o", "--output", help="Output file path")

    def execute(self, args):
        """Run the processing pipeline."""
        data = read_input(args.input)
        result = self.pipeline.run(data)
        if args.output:
            write_output(args.output, self.formatter.format(result))
        return result
