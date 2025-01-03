"""Constants for the RAG application.

This module contains truly constant values that never change during runtime.
These constants are fundamental to the application's operation and should
only be modified through code changes, not through configuration.

Constants are organized into categories:

1. Text Processing:
   - Separators used for text chunking and processing
   - These define how documents are split into manageable pieces

2. Model Constants:
   - Fixed model-related values
   - These are core to the application's ML functionality

Note: For configurable settings or environment-specific values:
- Use dynamic_settings.py for runtime-configurable settings
- Use settings.py for environment-specific configuration
"""

# Text processing constants
TEXT_SEPARATORS = [
    "\n\n",  # Paragraph breaks
    "\n",    # Line breaks
    ".",     # Sentence endings
    "!",     # Exclamations
    "?",     # Questions
    ";",     # Semi-colons
    ":",     # Colons
    " ",     # Spaces
    ""       # Empty string fallback
]

# Model constants
DEFAULT_TOKENIZER = "cl100k_base"
