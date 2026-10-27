"""Project-specific exceptions for the engineering foundation."""


class EconForgeError(Exception):
    """Base class for expected EconForge errors."""


class ConfigurationError(EconForgeError):
    """Raised when application configuration is invalid or unsupported."""


class CoreCapabilityUnavailable(EconForgeError):
    """Raised if a caller requests protected core work before authorization."""

