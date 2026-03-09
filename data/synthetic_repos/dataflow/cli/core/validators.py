"""Validation rules for data quality checks."""


class SchemaValidator:
    """Validate data against a JSON schema."""

    def __init__(self, schema_path):
        self.schema_path = schema_path

    def validate(self, data):
        raise NotImplementedError


class RuleValidator:
    """Validate data against custom business rules."""

    def __init__(self, rules):
        self.rules = rules

    def validate(self, data):
        raise NotImplementedError


class CompositeValidator:
    """Combine multiple validators."""

    def __init__(self, validators):
        self.validators = validators

    def validate(self, data):
        """Run all validators and collect results."""
        raise NotImplementedError
