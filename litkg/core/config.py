"""Configuration management for LitKG SDK."""

import os
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Configuration class for LitKG SDK."""

    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None

    # Graph Database Configuration
    neo4j_uri: Optional[str] = None
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None

    # Processing Configuration
    confidence_threshold: float = 0.7
    batch_size: int = 10
    max_workers: int = 4
    timeout: int = 300

    # Feature Flags
    enable_communities: bool = True
    enable_temporal: bool = True
    enable_parallel_retrieval: bool = True
    enable_human_loop: bool = True

    # Domain Configuration
    domain: str = "general"
    language: str = "auto"

    # Output Configuration
    output_format: str = "neo4j"
    export_path: Optional[str] = None

    # Advanced Options
    custom_entities: list = field(default_factory=list)
    custom_relations: list = field(default_factory=list)
    extraction_prompt: Optional[str] = None

    def __post_init__(self):
        """Initialize configuration with environment variables."""
        # Load API keys from environment
        if not self.api_key:
            if self.llm_provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.llm_provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.llm_provider == "google":
                self.api_key = os.getenv("GOOGLE_API_KEY")

        # Load Neo4j configuration from environment
        if not self.neo4j_uri:
            self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        if not self.neo4j_password:
            self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        # Set output path
        if not self.export_path:
            self.export_path = os.getcwd()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from file."""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        import json
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key and self.llm_provider in ["openai", "anthropic", "google"]:
            raise ValueError(f"API key required for {self.llm_provider}")

        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")


def create_config(**kwargs) -> Config:
    """Create configuration with optional overrides."""
    return Config(**kwargs)