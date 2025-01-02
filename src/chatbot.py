from typing import List
import httpx
from openai import OpenAI
from config.settings import OPENAI_API_KEY
from config.dynamic_settings import settings_manager

class Chatbot:
    def __init__(self):
        """Initialize the chatbot with OpenAI API key and response cache."""
        http_client = httpx.Client()
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=http_client
        )
        # Get initial settings
        self.settings = settings_manager.get_all_settings()
        
        # Register as observer for settings changes
        settings_manager.add_observer(self._handle_settings_change)
        
        # Cache for storing responses
        self._response_cache = {}

    def _handle_settings_change(self, setting_name: str, new_value: dict) -> None:
        """Handle settings changes from the settings manager."""
        if setting_name in ['llm', 'response']:
            self.settings[setting_name] = new_value
            # Clear cache when settings change
            self._response_cache.clear()

    def _get_cache_key(self, context: str, query: str) -> str:
        """Generate a deterministic cache key for responses."""
        # Normalize whitespace and case for consistent keys
        normalized_context = " ".join(context.strip().lower().split())
        normalized_query = " ".join(query.strip().lower().split())
        return f"{normalized_query}|||{normalized_context}"

    def generate_response(self, context: str, query: str) -> str:
        """
        Generate a response using OpenAI's API based on context and query.
        
        Args:
            context (str): Relevant context retrieved from the database
            query (str): User's query
            
        Returns:
            str: Generated response from the model
            
        Raises:
            Exception: If there's an error in generating the response
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(context, query)
            cached_response = self._response_cache.get(cache_key)
            if cached_response is not None:
                return cached_response

            messages = [
                {"role": "system", "content": self.settings['response']['system_prompt']},
                {"role": "user", "content": f"""
                Context:
                {context}

                Question:
                {query}

                Please provide a detailed and comprehensive answer based on the context above. Include relevant examples and explanations where appropriate.
                """}
            ]

            response = self.client.chat.completions.create(
                model=self.settings['llm']['model'],
                messages=messages,
                temperature=self.settings['llm']['temperature'],
                max_tokens=self.settings['llm']['max_tokens'],
                seed=42  # Fixed seed for consistent sampling
            )

            response_text = response.choices[0].message.content.strip()
            # Cache the response
            self._response_cache[cache_key] = response_text
            return response_text

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def _format_contexts_for_cache(self, contexts: List[dict]) -> str:
        """Format contexts list into a deterministic string for caching."""
        # Sort contexts by text to ensure consistent ordering
        sorted_contexts = sorted(contexts, key=lambda x: x['text'])
        return "\n\n".join([
            f"Source {i+1}:\n{ctx['text']}"
            for i, ctx in enumerate(sorted_contexts)
        ])

    def generate_response_with_sources(self, contexts: List[dict], query: str) -> str:
        """
        Generate a response with source citations using OpenAI's API.
        
        Args:
            contexts (List[dict]): List of context dictionaries with text and metadata
            query (str): User's query
            
        Returns:
            str: Response text with source citations
            
        Raises:
            Exception: If there's an error in generating the response
        """
        try:
            # Format contexts and generate cache key
            formatted_contexts = self._format_contexts_for_cache(contexts)
            cache_key = self._get_cache_key(formatted_contexts, query)
            
            # Check cache first
            cached_response = self._response_cache.get(cache_key)
            if cached_response is not None:
                return cached_response

            messages = [
                {"role": "system", "content": self.settings['response']['source_citation_prompt']},
                {"role": "user", "content": f"""
                Sources:
                {formatted_contexts}

                Question:
                {query}

                Please provide a comprehensive answer with source citations, including detailed explanations and examples where appropriate.
                """}
            ]

            response = self.client.chat.completions.create(
                model=self.settings['llm']['model'],
                messages=messages,
                temperature=self.settings['llm']['temperature'],
                max_tokens=self.settings['llm']['max_tokens'],
                seed=42  # Fixed seed for consistent sampling
            )

            response_text = response.choices[0].message.content.strip()
            # Cache the response
            self._response_cache[cache_key] = response_text
            return response_text

        except Exception as e:
            raise Exception(f"Error generating response with sources: {str(e)}")

    def __del__(self):
        """Clean up by removing observer when object is destroyed."""
        try:
            settings_manager.remove_observer(self._handle_settings_change)
        except:
            pass  # Ignore errors during cleanup
