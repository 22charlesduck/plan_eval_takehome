"""JSON output formatter."""

import json
from cli.formats.base import BaseFormatter


class JsonFormatter(BaseFormatter):
    """Format output as JSON."""

    def __init__(self, indent=2, sort_keys=False):
        self.indent = indent
        self.sort_keys = sort_keys

    def format(self, data):
        """Serialize data to a JSON string."""
        return json.dumps(data, indent=self.indent, sort_keys=self.sort_keys)

    def file_extension(self):
        return ".json"
