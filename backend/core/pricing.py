import httpx
from functools import lru_cache
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PricingService:
    _pricing_cache: Dict[str, Any] = {}

    @classmethod
    def load_pricing(cls):
        """Fetches latest pricing from OpenRouter."""
        try:
            response = httpx.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            data = response.json()["data"]
            for model in data:
                cls._pricing_cache[model["id"]] = {
                    "prompt": float(model["pricing"]["prompt"]),
                    "completion": float(model["pricing"]["completion"])
                }
            logger.info("Pricing data loaded successfully from OpenRouter.")
        except Exception as e:
            logger.error(f"Failed to load pricing: {e}")
            # Fallback to some defaults if needed, or just log error
            # We don't want to crash the app if pricing fails to load, 
            # but cost calc will be 0.

    @classmethod
    def calculate_cost(cls, model_id: str, input_tokens: int, output_tokens: int) -> float:
        if not cls._pricing_cache:
            cls.load_pricing()
          
        pricing = cls._pricing_cache.get(model_id)
        if not pricing:
            # Try to load again if missing (maybe new model or cache empty)
            cls.load_pricing()
            pricing = cls._pricing_cache.get(model_id)
            
        if not pricing:
            logger.warning(f"Pricing not found for model: {model_id}")
            return 0.0
              
        # Pricing is typically per 1 token or per 1M tokens depending on API.
        # OpenRouter API returns price *per token* usually, but verify normalization.
        # Based on docs/experience, OpenRouter pricing API returns cost per token (very small float).
        cost = (input_tokens * pricing["prompt"]) + (output_tokens * pricing["completion"])
        return cost
