"""Schema definitions and validation."""


class SchemaDefinition:
    """Represents a data schema with field types and constraints."""

    def __init__(self, fields=None, required=None):
        self.fields = fields or {}
        self.required = required or []

    def validate(self, record):
        """Validate a single record against this schema."""
        raise NotImplementedError


def validate_schema(data, schema_config):
    """Validate data against schema if provided, otherwise pass through."""
    if schema_config is None:
        return data
    schema = SchemaDefinition(**schema_config)
    for record in data:
        schema.validate(record)
    return data
