"""Entry point and argument parsing for the dataflow CLI."""

import argparse
import sys
from cli.commands.process import ProcessCommand
from cli.commands.validate import ValidateCommand
from cli.commands.inspect import InspectCommand
from cli.formats.json_output import JsonFormatter
from cli.formats.csv_output import CsvFormatter
from cli.utils.logging import setup_logging
from cli.utils.config import load_config


FORMAT_REGISTRY = {
    "json": JsonFormatter,
    "csv": CsvFormatter,
}

COMMAND_REGISTRY = {
    "process": ProcessCommand,
    "validate": ValidateCommand,
    "inspect": InspectCommand,
}


def build_parser():
    """Build the main argument parser."""
    parser = argparse.ArgumentParser(prog="dataflow", description="CLI data processing tool")
    parser.add_argument("--format", choices=list(FORMAT_REGISTRY.keys()), default="json")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    subparsers = parser.add_subparsers(dest="command")

    for name, cmd_cls in COMMAND_REGISTRY.items():
        sub = subparsers.add_parser(name, help=cmd_cls.help_text)
        cmd_cls.add_arguments(sub)

    return parser


def main(argv=None):
    """Main entry point."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    formatter = FORMAT_REGISTRY[args.format]()
    command = COMMAND_REGISTRY[args.command](config=config, formatter=formatter)
    result = command.execute(args)
    print(formatter.format(result))


if __name__ == "__main__":
    main()
