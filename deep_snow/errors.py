class ModelCompatibilityError(RuntimeError):
    """Raised when a checkpoint does not match the current model input schema."""


class AcquisitionError(RuntimeError):
    """Raised when prediction inputs cannot be acquired."""


class EmptyAcquisitionError(AcquisitionError):
    """Raised when no usable acquisitions are found for a required data source."""


class TransientAcquisitionError(AcquisitionError):
    """Raised when a temporary acquisition failure is likely retryable."""
