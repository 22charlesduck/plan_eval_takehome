"""Abstract base formatter."""

from abc import ABC, abstractmethod


class BaseFormatter(ABC):
    """Base class for all output formatters."""

    @abstractmethod
    def format(self, data):
        """Format the given data into a string representation."""
        raise NotImplementedError

    @abstractmethod
    def file_extension(self):
        """Return the default file extension for this format."""
        raise NotImplementedError
