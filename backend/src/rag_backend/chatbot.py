from typing import List
import httpx
from openai import OpenAI
from rag_backend.config.settings import OPENAI_API_KEY
from rag_backend.config.dynamic_settings import settings_manager

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
        # Group contexts by source
        contexts_by_source = {}
        for ctx in contexts:
            source = ctx.get('source', 'Unknown')
            if source not in contexts_by_source:
                contexts_by_source[source] = []
            contexts_by_source[source].append(ctx)
        
        # Sort sources and their contexts
        formatted_parts = []
        for source in sorted(contexts_by_source.keys()):
            source_contexts = contexts_by_source[source]
            # Sort contexts within each source by chunk index
            source_contexts.sort(key=lambda x: x.get('chunk_index', 0))
            
            # Format contexts for this source
            source_text = f"Source: {source}\n"
            source_text += "Title: " + source_contexts[0].get('title', 'Untitled') + "\n"
            source_text += "Content:\n"
            source_text += "\n".join([
                f"[Chunk {ctx.get('chunk_index', 0)+1}/{ctx.get('total_chunks', 1)}] {ctx['text']}"
                for ctx in source_contexts
            ])
            formatted_parts.append(source_text)
        
        return "\n\n".join(formatted_parts)

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

            # Get unique source names for the overview
            unique_sources = sorted(set(ctx['source'] for ctx in contexts))
            source_overview = "\n".join([
                f"* [Source {i+1}: {source}]" 
                for i, source in enumerate(unique_sources)
            ])

            messages = [
                {"role": "system", "content": self.settings['response']['source_citation_prompt']},
                {"role": "user", "content": f"""
                Source Overview:
                {source_overview}

                Source Details:
                {formatted_contexts}

                Question:
                {query}

                Please provide a comprehensive answer that synthesizes information across all sources. Remember to:
                1. Start with the source overview list
                2. Compare and contrast information from different sources
                3. Organize information thematically rather than source-by-source
                4. Note any agreements or disagreements between sources
                5. Maintain balanced representation from all sources
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
