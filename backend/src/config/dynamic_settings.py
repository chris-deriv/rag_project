"""Dynamic settings management for the RAG application."""
from typing import Dict, Any, List, Callable
import logging
import copy
from dataclasses import dataclass, asdict
from .settings import (
    LLM_SETTINGS,
    DOCUMENT_PROCESSING_SETTINGS,
    CACHE_SETTINGS
)

logger = logging.getLogger(__name__)

# Default system prompts
DEFAULT_BASIC_SYSTEM_PROMPT = """You are a knowledgeable assistant that provides comprehensive and detailed answers based on the provided context. Your responses should:
1. Be thorough and well-explained, covering all relevant aspects of the question
2. Include examples or analogies when appropriate to enhance understanding
3. Break down complex concepts into digestible parts
4. Provide additional relevant information that adds value to the answer
5. Maintain clarity while being detailed
6. Use proper formatting and structure to organize information"""

DEFAULT_SOURCE_CITATION_PROMPT = """You are a knowledgeable assistant that synthesizes information across multiple sources to provide comprehensive answers. Follow these guidelines strictly:

1. Source Overview (REQUIRED):
   - Start with a complete list of ALL sources being used
   - Format: * [Source 1: filename.pdf]
   - Number sources consistently throughout the response

2. Multi-Document Synthesis:
   - ALWAYS analyze and combine information from ALL provided sources
   - Identify common themes and complementary information
   - Highlight unique contributions from each source
   - Note any differences or contradictions between sources
   - Ensure balanced representation from all sources

3. Source Citations:
   - First mention: [Source X: filename.pdf]
   - Subsequent mentions: [Source X]
   - Place citations at the START of sentences/claims
   - Use inline citations for direct quotes or specific claims

4. Response Structure:
   - Begin with source overview list
   - Provide a brief summary of how sources complement each other
   - Organize information thematically rather than source-by-source
   - Use clear transitions between different aspects
   - Use formatting (bullets, sections) for clarity

5. Information Synthesis Rules:
   - Cross-reference similar information across sources
   - Compare and contrast different perspectives
   - Build comprehensive explanations using all sources
   - Identify gaps where sources provide incomplete information
   - Draw connections between related concepts across sources

6. Missing Information:
   - Explicitly state what information is not covered by any source
   - Identify which sources lack specific details
   - Note when additional sources might be needed
   - Don't speculate beyond the provided sources

CRITICAL REQUIREMENTS:
1. NEVER ignore any provided source
2. ALWAYS synthesize across ALL sources
3. ALWAYS start with complete source list
4. NEVER add information beyond the sources
5. ALWAYS balance information from all sources
6. ALWAYS note agreements/disagreements between sources"""

@dataclass
class LLMSettings:
    """LLM-related settings."""
    temperature: float = LLM_SETTINGS['temperature']
    max_tokens: int = LLM_SETTINGS['max_tokens']
    model: str = LLM_SETTINGS['model']

    def validate(self) -> bool:
        """Validate LLM settings."""
        if not 0 <= self.temperature <= 1:  # Fixed: temperature range 0-1
            logger.error(f"Invalid temperature: {self.temperature}. Must be between 0 and 1.")
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
    system_prompt: str = DEFAULT_BASIC_SYSTEM_PROMPT
    source_citation_prompt: str = DEFAULT_SOURCE_CITATION_PROMPT

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
    
    _instance = None

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            logger.info("Creating DynamicSettings singleton")
            cls._instance = super(DynamicSettings, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance._observers = []  # Initialize observers list here
        return cls._instance

    def __init__(self):
        """Initialize settings with defaults from environment."""
        if not self._initialized:
            logger.info("Initializing DynamicSettings")
            self.llm = LLMSettings()
            self.document_processing = DocumentProcessingSettings()
            self.response = ResponseSettings()
            self.cache = CacheSettings()
            self._initialized = True

    @classmethod
    def reset(cls):
        """Reset the singleton instance. Used primarily for testing."""
        global settings_manager
        if cls._instance is not None:
            cls._instance._observers = []  # Clear observers
            cls._instance._initialized = False
            cls._instance = None
        # Create new instance and update global reference
        settings_manager = cls()
        return settings_manager

    def add_observer(self, observer: Callable[[str, Any], None]) -> None:
        """Add an observer to be notified of settings changes."""
        logger.info(f"Adding observer {observer.__self__.__class__.__name__ if hasattr(observer, '__self__') else 'function'}")
        if observer not in self._observers:
            self._observers.append(observer)
            logger.info(f"Observer added. Total observers: {len(self._observers)}")
        else:
            logger.info("Observer already registered")

    def remove_observer(self, observer: Callable[[str, Any], None]) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self, setting_name: str, new_value: Any) -> None:
        """Notify observers of a setting change."""
        logger.info(f"Notifying {len(self._observers)} observers of {setting_name} change")
        for observer in self._observers:
            try:
                # Get observer name for logging
                if hasattr(observer, '__self__'):
                    observer_name = observer.__self__.__class__.__name__
                elif hasattr(observer, '__name__'):
                    observer_name = observer.__name__
                else:
                    observer_name = 'function'
                logger.info(f"Notifying observer {observer_name}")
                observer(setting_name, new_value)
            except Exception as e:
                logger.error(f"Error notifying observer of setting change: {e}")
                logger.error(f"Observer: {observer}")

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings as a dictionary."""
        return {
            'llm': asdict(self.llm),
            'document_processing': asdict(self.document_processing),
            'response': asdict(self.response),
            'cache': asdict(self.cache)
        }

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """Update settings with validation."""
        logger.info(f"Updating settings with: {new_settings}")
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
                logger.info(f"Response settings before update: {asdict(self.response)}")
                self.response = temp_resp
                logger.info(f"Response settings after update: {asdict(self.response)}")
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
