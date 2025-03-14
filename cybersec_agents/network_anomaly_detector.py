import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelType, RoleType

from .config import Config
from .exceptions import FileError, ValidationError
from .utils.file_validator import FileValidator
from .utils.rate_limiter import RateLimiter


class NetworkAnomalyDetector:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.agent.rate_limit, time_window=60
        )
        self.file_validator = FileValidator(
            allowed_extensions=self.config.security.allowed_file_types,
            max_size_mb=self.config.security.max_file_size_mb,
        )
        self.logger = logging.getLogger(__name__)

        # Initialize CAMEL agent
        model = ModelFactory.create(model_type=ModelType.GPT_4)
        system_message = BaseMessage(
            role_name="Security Analyst",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="You are a cybersecurity expert specialized in network security and anomaly detection.",
        )
        self.agent = ChatAgent(system_message=system_message, model=model)

    async def analyze_nmap_output(
        self, nmap_file: Union[str, Path]
    ) -> Dict[str, List[dict]]:
        """Analyzes Nmap scan results for security insights.

        Args:
            nmap_file: Path to the Nmap output file

        Returns:
            Dictionary containing analysis results

        Raises:
            FileError: If file validation fails
            SecurityError: If security checks fail
            ValidationError: If input data is invalid
        """
        nmap_path = Path(nmap_file)

        # Validate file
        try:
            self.file_validator.validate(nmap_path)
        except ValidationError as e:
            self.logger.error(f"File validation failed: {e}")
            raise FileError(str(nmap_path), str(e))

        # Rate limiting
        await self.rate_limiter.acquire()

        try:
            # Read and analyze file
            with open(nmap_path, "r") as f:
                nmap_data = f.read()

            analysis_result = await self._analyze_data(nmap_data)

            # Log success
            self.logger.info(f"Successfully analyzed {nmap_path} at {datetime.now()}")

            return analysis_result

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    async def _analyze_data(self, data: str) -> Dict[str, List[dict]]:
        """Analyzes network data using CAMEL agent.

        Args:
            data: Raw data to analyze

        Returns:
            Structured analysis results
        """
        prompt = f"""
        Analyze this network scan data and identify:
        1. Potential security threats
        2. Vulnerabilities
        3. Recommended actions
        4. Risk level assessment

        Data:
        {data}
        
        Provide the analysis in a structured format.
        """

        await self.agent.chat(prompt)

        # Parse and structure the response
        # This is a simplified example - you might want to add more sophisticated parsing
        results = {
            "threats": [],
            "vulnerabilities": [],
            "recommendations": [],
            "risk_level": "UNKNOWN",
        }

        # Validate results
        self._validate_results(results)

        return results

    def _validate_results(self, results: Dict[str, List[dict]]) -> bool:
        """Validates analysis results before returning.

        Args:
            results: Analysis results to validate

        Returns:
            True if valid, raises ValidationError otherwise
        """
        required_keys = {"threats", "recommendations", "risk_level"}
        if not all(key in results for key in required_keys):
            raise ValidationError("Missing required keys in analysis results")
        return True
