from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    model_name: str = Field(default="gpt-4", description="AI model to use")
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY", description="Environment variable for OpenAI API key"
    )
    rate_limit: int = Field(default=60, description="Maximum requests per minute")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")


class SecurityConfig(BaseModel):
    allowed_file_types: list[str] = Field(
        default=["txt", "pcap", "xml", "json"], description="Allowed file extensions"
    )
    max_file_size_mb: int = Field(default=10, description="Maximum file size in MB")
    scan_timeout: int = Field(
        default=300, description="Maximum scan duration in seconds"
    )


class Config(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    log_level: str = Field(default="INFO")
    output_dir: Path = Field(default=Path("./output"))

    @classmethod
    def load_from_file(cls, config_path: Path) -> "Config":
        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls.parse_obj(config_data)
