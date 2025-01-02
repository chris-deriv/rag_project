"""Dynamic settings management for the RAG application."""
from typing import Dict, Any, List, Callable
import logging
from dataclasses import dataclass, asdict
from .constants import BASIC_SYSTEM_PROMPT, SOURCE_CITATION_PROMPT
from .settings import (
    LLM_SETTINGS,
    DOCUMENT_PROCESSING_SETTINGS,
    CACHE_SETTINGS
)

logger = logging.getLogger(__name__)

@dataclass
class LLMSettings:
    """LLM-related settings."""
    temperature: float = LLM_SETTINGS['temperature']
    max_tokens: int = LLM_SETTINGS['max_tokens']
    model: str = LLM_SETTINGS['model']

    def validate(self) -> bool:
        """Validate LLM settings."""
        if not 0 <= self.temperature <= 2:
            logger.error(f"Invalid temperature: {self.temperature}. Must be between 0 and 2.")
            return False
        if self.max_tokens < 1:
            logger.error(f"Invalid max_tokens: {self.max_tokens}. Must be positive.")
            return False
        return True

@dataclass
class DocumentProcessingSettings:
    """Document processing settings."""
    chunk_size: int = DOCUMENT_PROCESSING_SETTINGS['chunk_size']
    chunk_overlap: int = DOCUMENT_PROCESSING_SETTINGS['chunk_overlap']

    def validate(self) -> bool:
        """Validate document processing settings."""
        if self.chunk_size < 100:
            logger.error(f"Invalid chunk_size: {self.chunk_size}. Must be at least 100.")
            return False
        if self.chunk_overlap >= self.chunk_size:
            logger.error(f"Invalid chunk_overlap: {self.chunk_overlap}. Must be less than chunk_size.")
            return False
        return True

@dataclass
class ResponseSettings:
    """Response generation settings."""
    system_prompt: str = BASIC_SYSTEM_PROMPT
    source_citation_prompt: str = SOURCE_CITATION_PROMPT

    def validate(self) -> bool:
        """Validate response settings."""
        if not self.system_prompt.strip():
            logger.error("System prompt cannot be empty.")
            return False
        if not self.source_citation_prompt.strip():
            logger.error("Source citation prompt cannot be empty.")
            return False
        return True

@dataclass
class CacheSettings:
    """Cache settings."""
    enabled: bool = CACHE_SETTINGS['enabled']
    size: int = CACHE_SETTINGS['size']

    def validate(self) -> bool:
        """Validate cache settings."""
        if self.size < 1:
            logger.error(f"Invalid cache size: {self.size}. Must be positive.")
            return False
        return True

class DynamicSettings:
    """Manages dynamic settings with validation and change notification."""
    
    def __init__(self):
        """Initialize settings with defaults from environment."""
        self.llm = LLMSettings()
        self.document_processing = DocumentProcessingSettings()
        self.response = ResponseSettings()
        self.cache = CacheSettings()
        self._observers: List[Callable[[str, Any], None]] = []

    def add_observer(self, observer: Callable[[str, Any], None]) -> None:
        """Add an observer to be notified of settings changes."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[str, Any], None]) -> None:
        """Remove an observer."""
        self._observers.remove(observer)

    def _notify_observers(self, setting_name: str, new_value: Any) -> None:
        """Notify observers of a setting change."""
        for observer in self._observers:
            try:
                observer(setting_name, new_value)
            except Exception as e:
                logger.error(f"Error notifying observer of setting change: {e}")

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings as a dictionary."""
        return {
            'llm': asdict(self.llm),
            'document_processing': asdict(self.document_processing),
            'response': asdict(self.response),
            'cache': asdict(self.cache)
        }

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        Update settings with validation.
        
        Args:
            new_settings: Dictionary of settings to update
            
        Returns:
            bool: True if all updates were successful
        """
        success = True
        
        # Update LLM settings
        if 'llm' in new_settings:
            llm_settings = new_settings['llm']
            temp_llm = LLMSettings(
                temperature=llm_settings.get('temperature', self.llm.temperature),
                max_tokens=llm_settings.get('max_tokens', self.llm.max_tokens),
                model=llm_settings.get('model', self.llm.model)
            )
            if temp_llm.validate():
                self.llm = temp_llm
                self._notify_observers('llm', asdict(self.llm))
            else:
                success = False

        # Update document processing settings
        if 'document_processing' in new_settings:
            doc_settings = new_settings['document_processing']
            temp_doc = DocumentProcessingSettings(
                chunk_size=doc_settings.get('chunk_size', self.document_processing.chunk_size),
                chunk_overlap=doc_settings.get('chunk_overlap', self.document_processing.chunk_overlap)
            )
            if temp_doc.validate():
                self.document_processing = temp_doc
                self._notify_observers('document_processing', asdict(self.document_processing))
            else:
                success = False

        # Update response settings
        if 'response' in new_settings:
            resp_settings = new_settings['response']
            temp_resp = ResponseSettings(
                system_prompt=resp_settings.get('system_prompt', self.response.system_prompt),
                source_citation_prompt=resp_settings.get('source_citation_prompt', self.response.source_citation_prompt)
            )
            if temp_resp.validate():
                self.response = temp_resp
                self._notify_observers('response', asdict(self.response))
            else:
                success = False

        # Update cache settings
        if 'cache' in new_settings:
            cache_settings = new_settings['cache']
            temp_cache = CacheSettings(
                enabled=cache_settings.get('enabled', self.cache.enabled),
                size=cache_settings.get('size', self.cache.size)
            )
            if temp_cache.validate():
                self.cache = temp_cache
                self._notify_observers('cache', asdict(self.cache))
            else:
                success = False

        return success

# Global settings instance
settings_manager = DynamicSettings()
