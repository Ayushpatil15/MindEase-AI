"""AI Provider Abstraction — strategy pattern for optional LLM integration.

The core application works without any LLM (AI_PROVIDER=none).
When configured, a provider can enhance recommendation wording.
The deterministic fallback is always the safety net.

Configuration (via .env):
    AI_PROVIDER=none      →  NoneProvider (passthrough, default)
    AI_PROVIDER=gemini    →  GeminiProvider (requires GEMINI_API_KEY)
    AI_PROVIDER=mistral   →  MistralProvider (requires MISTRAL_API_KEY)

Safety:
    - API keys from env vars only — never in code, templates, or frontend.
    - Provider failure → logged warning → deterministic fallback returned.
    - AI is for wording enhancement only — never generates clinical advice.
"""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AIProvider(ABC):
    """Abstract base for AI text-enhancement providers."""

    @abstractmethod
    def enhance_text(self, text, context=None):
        """Enhance the given text, optionally using context.

        Args:
            text: The deterministic base text to enhance.
            context: Optional dict with user state context for personalization.

        Returns:
            str: Enhanced text, or the original text if enhancement fails.
        """
        pass

    @abstractmethod
    def is_available(self):
        """Return True if this provider is configured and ready."""
        pass


class NoneProvider(AIProvider):
    """Passthrough provider — returns text unchanged. Always available."""

    def enhance_text(self, text, context=None):
        return text

    def is_available(self):
        return True


class GeminiProvider(AIProvider):
    """Google Gemini provider — enhances text via the Gemini API.

    Requires GEMINI_API_KEY environment variable.
    Full API integration in Phase 6.
    """

    def __init__(self):
        self.api_key = os.environ.get('GEMINI_API_KEY')

    def enhance_text(self, text, context=None):
        if not self.is_available():
            logger.warning('GeminiProvider called but API key is not configured.')
            return text
        # Phase 6: actual Gemini API call
        # On failure, return the original deterministic text
        return text

    def is_available(self):
        return bool(self.api_key)


class MistralProvider(AIProvider):
    """Mistral AI provider — enhances text via the Mistral API.

    Requires MISTRAL_API_KEY environment variable.
    Full API integration in Phase 6.
    """

    def __init__(self):
        self.api_key = os.environ.get('MISTRAL_API_KEY')

    def enhance_text(self, text, context=None):
        if not self.is_available():
            logger.warning('MistralProvider called but API key is not configured.')
            return text
        # Phase 6: actual Mistral API call
        # On failure, return the original deterministic text
        return text

    def is_available(self):
        return bool(self.api_key)


# ── Provider Factory ──────────────────────────────────────────────────

_PROVIDERS = {
    'none': NoneProvider,
    'gemini': GeminiProvider,
    'mistral': MistralProvider,
}


def get_ai_provider(provider_name=None):
    """Instantiate and return the configured AI provider.

    Args:
        provider_name: Override for the AI_PROVIDER env var.

    Returns:
        An AIProvider instance. Defaults to NoneProvider if the
        requested provider is unknown.
    """
    if provider_name is None:
        provider_name = os.environ.get('AI_PROVIDER', 'none')

    provider_class = _PROVIDERS.get(provider_name.lower(), NoneProvider)

    if provider_class is not NoneProvider:
        logger.info('AI provider: %s', provider_name)
    return provider_class()
