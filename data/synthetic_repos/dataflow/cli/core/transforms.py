"""Data transformation functions."""


TRANSFORM_REGISTRY = {}


def register_transform(name):
    """Decorator to register a transform function."""
    def decorator(func):
        TRANSFORM_REGISTRY[name] = func
        return func
    return decorator


@register_transform("uppercase")
def uppercase_transform(data, **kwargs):
    """Convert string fields to uppercase."""
    raise NotImplementedError


@register_transform("filter")
def filter_transform(data, **kwargs):
    """Filter records based on conditions."""
    raise NotImplementedError


@register_transform("aggregate")
def aggregate_transform(data, **kwargs):
    """Aggregate records by key."""
    raise NotImplementedError


def apply_transforms(data, transform_specs):
    """Apply a sequence of transforms to the data."""
    result = data
    for spec in transform_specs:
        name = spec["name"]
        params = spec.get("params", {})
        if name not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown transform: {name}")
        result = TRANSFORM_REGISTRY[name](result, **params)
    return result
