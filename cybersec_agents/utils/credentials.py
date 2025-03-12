import os
from dataclasses import dataclass


@dataclass
class CredentialManager:
    """Manages API credentials with fallback to environment variables."""

    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self._secret_client = None

    @property
    def secret_client(self):
        """Lazy load the secret client only when needed."""
        if self._secret_client is None and self.project_id:
            try:
                from google.cloud import secretmanager

                self._secret_client = secretmanager.SecretManagerServiceClient()
            except ImportError:
                print(
                    "Google Cloud Secret Manager not available. Using environment variables."
                )
        return self._secret_client

    def get_credential(self, key_name: str) -> str:
        """Retrieve API credentials from GCP Secret Manager or environment variables.

        Args:
            key_name: Name of the credential to retrieve

        Returns:
            str: The credential value
        """
        # First try environment variable
        env_value = os.getenv(key_name.upper())
        if env_value:
            return env_value

        # Then try Google Cloud Secret Manager if available
        if self.secret_client and self.project_id:
            try:
                name = f"projects/{self.project_id}/secrets/{key_name}/versions/latest"
                response = self.secret_client.access_secret_version(
                    request={"name": name}
                )
                return response.payload.data.decode("UTF-8")
            except Exception as e:
                print(f"Failed to get credential from Secret Manager: {e}")
                return ""

        return ""

    def rotate_credentials(self, key_name: str, new_value: str) -> bool:
        """Rotate API credentials.

        Args:
            key_name: Name of the credential to rotate
            new_value: New credential value

        Returns:
            bool: True if rotation was successful
        """
        if not self.secret_client or not self.project_id:
            print("Secret Manager not configured. Cannot rotate credentials.")
            return False

        try:
            parent = f"projects/{self.project_id}/secrets/{key_name}"
            payload = new_value.encode("UTF-8")

            self.secret_client.add_secret_version(
                request={"parent": parent, "payload": {"data": payload}}
            )
            return True
        except Exception as e:
            print(f"Failed to rotate credential: {e}")
            return False
