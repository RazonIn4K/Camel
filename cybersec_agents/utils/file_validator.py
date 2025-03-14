from pathlib import Path
from typing import List, Set

from ..exceptions import ValidationError


class FileValidator:
    def __init__(self, allowed_extensions: List[str], max_size_mb: int):
        self.allowed_extensions: Set[str] = set(allowed_extensions)
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def validate(self, file_path: Path) -> None:
        """Validates a file against security rules.

        Args:
            file_path: Path to file to validate

        Raises:
            ValidationError: If validation fails
        """
        # Check if file exists
        if not file_path.exists():
            raise ValidationError(f"File {file_path} does not exist")

        # Check extension
        if file_path.suffix[1:] not in self.allowed_extensions:
            raise ValidationError(
                f"File type {file_path.suffix} not allowed. "
                f"Allowed types: {', '.join(self.allowed_extensions)}"
            )

        # Check file size
        if file_path.stat().st_size > self.max_size_bytes:
            raise ValidationError(
                f"File size exceeds maximum allowed size of {self.max_size_bytes/1024/1024}MB"
            )

        # Basic security checks
        self._security_check(file_path)

    def _security_check(self, file_path: Path) -> None:
        """Performs additional security checks on the file.

        Args:
            file_path: Path to file to check

        Raises:
            ValidationError: If security check fails
        """
        # Check for symbolic links
        if file_path.is_symlink():
            raise ValidationError("Symbolic links are not allowed")

        # Check for absolute path traversal
        if file_path.is_absolute():
            raise ValidationError("Absolute paths are not allowed")

        # Check for relative path traversal
        if ".." in file_path.parts:
            raise ValidationError("Path traversal detected")
