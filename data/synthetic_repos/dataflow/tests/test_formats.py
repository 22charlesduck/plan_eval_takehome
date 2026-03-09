"""Tests for output formatters."""

import pytest
from cli.formats.json_output import JsonFormatter
from cli.formats.csv_output import CsvFormatter


class TestJsonFormatter:
    def test_format_dict(self):
        fmt = JsonFormatter()
        result = fmt.format({"key": "value"})
        assert '"key"' in result

    def test_format_list(self, sample_data):
        fmt = JsonFormatter()
        result = fmt.format(sample_data)
        assert "Alice" in result


class TestCsvFormatter:
    def test_format_records(self, sample_data):
        fmt = CsvFormatter()
        result = fmt.format(sample_data)
        assert "name" in result
