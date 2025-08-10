"""
EXPLAINIUM - Custom Exceptions

Custom exception classes for the application to handle different types of errors
in a structured and informative way.
"""


class ExplainiumError(Exception):
    """Base exception class for all Explainium errors"""
    pass


class ProcessingError(ExplainiumError):
    """Raised when document processing fails"""
    pass


class AIError(ExplainiumError):
    """Raised when AI operations fail"""
    pass


class DatabaseError(ExplainiumError):
    """Raised when database operations fail"""
    pass


class ValidationError(ExplainiumError):
    """Raised when input validation fails"""
    pass


class ConfigurationError(ExplainiumError):
    """Raised when configuration is invalid or missing"""
    pass


class FileError(ExplainiumError):
    """Raised when file operations fail"""
    pass


class NetworkError(ExplainiumError):
    """Raised when network operations fail"""
    pass


class AuthenticationError(ExplainiumError):
    """Raised when authentication fails"""
    pass


class AuthorizationError(ExplainiumError):
    """Raised when authorization fails"""
    pass


class RateLimitError(ExplainiumError):
    """Raised when rate limits are exceeded"""
    pass


class TimeoutError(ExplainiumError):
    """Raised when operations timeout"""
    pass