from urllib.error import HTTPError, URLError


class ModelCompatibilityError(RuntimeError):
    """Raised when a checkpoint does not match the current model input schema."""


class AcquisitionError(RuntimeError):
    """Raised when prediction inputs cannot be acquired."""


class EmptyAcquisitionError(AcquisitionError):
    """Raised when no usable acquisitions are found for a required data source."""


class TransientAcquisitionError(AcquisitionError):
    """Raised when a temporary acquisition failure is likely retryable."""


def is_likely_transient_error(exc):
    message = str(exc).lower()
    transient_markers = [
        "timeout",
        "timed out",
        "maximum allowed time",
        "please try again",
        "temporarily unavailable",
        "temporary failure",
        "connection reset",
        "connection aborted",
        "connection refused",
        "remote disconnected",
        "service unavailable",
        "too many requests",
        "502",
        "503",
        "504",
    ]
    return isinstance(exc, (TimeoutError, ConnectionError, HTTPError, URLError)) or any(
        marker in message for marker in transient_markers
    )
