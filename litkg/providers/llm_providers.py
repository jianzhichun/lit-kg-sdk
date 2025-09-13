"""LLM provider integrations for LitKG SDK."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    metadata: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    async def extract_structured(self, prompt: str) -> Dict[str, Any]:
        """Extract structured data from text."""
        pass

    @abstractmethod
    async def generate(self, prompt: str) -> LLMResponse:
        """Generate text response."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        except ImportError:
            logger.error("OpenAI package not found. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self.client is not None and self.config.api_key is not None

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate text using OpenAI."""
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available")

        try:
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                metadata={"model": self.config.llm_model},
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def extract_structured(self, prompt: str) -> Dict[str, Any]:
        """Extract structured data using OpenAI."""
        # Add JSON formatting instruction
        structured_prompt = f"""
{prompt}

Please respond with valid JSON only. Do not include any additional text or explanations.
"""

        response = await self.generate(structured_prompt)

        try:
            # Parse JSON response
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from OpenAI response: {e}")
            # Try to extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                logger.error(f"Could not extract valid JSON from: {content}")
                return {"entities": [], "relations": []}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, config):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        except ImportError:
            logger.error("Anthropic package not found. Install with: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")

    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return self.client is not None and self.config.api_key is not None

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate text using Anthropic."""
        if not self.is_available():
            raise RuntimeError("Anthropic provider not available")

        try:
            response = await self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens or 4000,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return LLMResponse(
                content=response.content[0].text,
                metadata={"model": self.config.llm_model},
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def extract_structured(self, prompt: str) -> Dict[str, Any]:
        """Extract structured data using Anthropic."""
        structured_prompt = f"""
{prompt}

Please respond with valid JSON only. Do not include any additional text or explanations.
"""

        response = await self.generate(structured_prompt)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Anthropic response: {e}")
            # Similar cleanup as OpenAI
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                logger.error(f"Could not extract valid JSON from: {content}")
                return {"entities": [], "relations": []}


class GoogleProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, config):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Google client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self.client = genai.GenerativeModel(self.config.llm_model)
        except ImportError:
            logger.error("Google AI package not found. Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Google client: {e}")

    def is_available(self) -> bool:
        """Check if Google is available."""
        return self.client is not None and self.config.api_key is not None

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate text using Google."""
        if not self.is_available():
            raise RuntimeError("Google provider not available")

        try:
            # Google doesn't have native async, so run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(
                    prompt,
                    generation_config={"temperature": self.config.temperature}
                )
            )

            return LLMResponse(
                content=response.text,
                metadata={"model": self.config.llm_model},
                usage=None  # Google doesn't provide detailed usage info
            )

        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise

    async def extract_structured(self, prompt: str) -> Dict[str, Any]:
        """Extract structured data using Google."""
        structured_prompt = f"""
{prompt}

Please respond with valid JSON only. Do not include any additional text or explanations.
"""

        response = await self.generate(structured_prompt)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Google response: {e}")
            # Similar cleanup
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                logger.error(f"Could not extract valid JSON from: {content}")
                return {"entities": [], "relations": []}


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self.client = ollama.AsyncClient()
        except ImportError:
            logger.error("Ollama package not found. Install with: pip install ollama")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self.client is not None

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate text using Ollama."""
        if not self.is_available():
            raise RuntimeError("Ollama provider not available")

        try:
            response = await self.client.chat(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens or -1
                }
            )

            return LLMResponse(
                content=response['message']['content'],
                metadata={"model": self.config.llm_model},
                usage=None  # Ollama doesn't provide usage stats
            )

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    async def extract_structured(self, prompt: str) -> Dict[str, Any]:
        """Extract structured data using Ollama."""
        structured_prompt = f"""
{prompt}

Please respond with valid JSON only. Do not include any additional text or explanations.
Format your response as a JSON object with 'entities' and 'relations' arrays.
"""

        response = await self.generate(structured_prompt)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Ollama response: {e}")
            # More aggressive cleanup for Ollama
            content = response.content.strip()

            # Remove code blocks
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    if part.strip().startswith("{") and part.strip().endswith("}"):
                        content = part.strip()
                        break

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Could not extract valid JSON from: {content}")
                return {"entities": [], "relations": []}


class LiteLLMProvider(LLMProvider):
    """LiteLLM unified provider for multiple LLMs."""

    def __init__(self, config):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize LiteLLM."""
        try:
            import litellm
            self.client = litellm
            # Set API key based on provider
            if self.config.api_key:
                litellm.api_key = self.config.api_key
        except ImportError:
            logger.error("LiteLLM package not found. Install with: pip install litellm")
        except Exception as e:
            logger.error(f"Failed to initialize LiteLLM: {e}")

    def is_available(self) -> bool:
        """Check if LiteLLM is available."""
        return self.client is not None

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate text using LiteLLM."""
        if not self.is_available():
            raise RuntimeError("LiteLLM provider not available")

        try:
            response = await self.client.acompletion(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                metadata={"model": self.config.llm_model},
                usage=getattr(response, 'usage', None)
            )

        except Exception as e:
            logger.error(f"LiteLLM API error: {e}")
            raise

    async def extract_structured(self, prompt: str) -> Dict[str, Any]:
        """Extract structured data using LiteLLM."""
        structured_prompt = f"""
{prompt}

Please respond with valid JSON only. Do not include any additional text or explanations.
"""

        response = await self.generate(structured_prompt)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LiteLLM response: {e}")
            return {"entities": [], "relations": []}


def get_llm_provider(config) -> LLMProvider:
    """Get appropriate LLM provider based on configuration."""

    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
        "litellm": LiteLLMProvider
    }

    provider_class = provider_map.get(config.llm_provider.lower())
    if not provider_class:
        # Try LiteLLM as fallback for unknown providers
        logger.warning(f"Unknown provider '{config.llm_provider}', trying LiteLLM")
        provider_class = LiteLLMProvider

    provider = provider_class(config)

    # Check if provider is available
    if not provider.is_available():
        logger.error(f"Provider '{config.llm_provider}' is not available")
        # Try to suggest fallbacks
        available_providers = []
        for name, cls in provider_map.items():
            test_provider = cls(config)
            if test_provider.is_available():
                available_providers.append(name)

        if available_providers:
            logger.info(f"Available providers: {', '.join(available_providers)}")
        else:
            logger.error("No LLM providers are available. Please check your API keys and dependencies.")

    return provider