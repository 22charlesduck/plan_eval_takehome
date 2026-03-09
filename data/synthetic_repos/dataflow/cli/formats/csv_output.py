"""CSV output formatter."""

import csv
import io
from cli.formats.base import BaseFormatter


class CsvFormatter(BaseFormatter):
    """Format output as CSV."""

    def __init__(self, delimiter=","):
        self.delimiter = delimiter

    def format(self, data):
        """Serialize tabular data to CSV string."""
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys(), delimiter=self.delimiter)
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()

    def file_extension(self):
        return ".csv"
