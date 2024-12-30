from typing import List
import httpx
from openai import OpenAI
from config.settings import OPENAI_API_KEY

class Chatbot:
    def __init__(self):
        """Initialize the chatbot with OpenAI API key and response cache."""
        http_client = httpx.Client()
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=http_client
        )
        self.model = "gpt-3.5-turbo"  # Using the more modern chat model
        # Cache for storing responses
        self._response_cache = {}

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
                {"role": "system", "content": """You are a knowledgeable assistant that provides comprehensive and detailed answers based on the provided context. Your responses should:
                1. Be thorough and well-explained, covering all relevant aspects of the question
                2. Include examples or analogies when appropriate to enhance understanding
                3. Break down complex concepts into digestible parts
                4. Provide additional relevant information that adds value to the answer
                5. Maintain clarity while being detailed
                6. Use proper formatting and structure to organize information
                """},
                {"role": "user", "content": f"""
                Context:
                {context}

                Question:
                {query}

                Please provide a detailed and comprehensive answer based on the context above. Include relevant examples and explanations where appropriate.
                """}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Slightly increased for more natural responses while maintaining consistency
                max_tokens=1000,  # Increased to allow for more detailed responses
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
                {"role": "system", "content": """
                You are a knowledgeable assistant that provides comprehensive answers based on provided sources. For each response:
                1. Use information only from the provided sources
                2. Cite sources using [Source X] notation
                3. Provide detailed explanations and elaborate on key points
                4. Include relevant examples or analogies when appropriate
                5. Break down complex information into clear, digestible sections
                6. Synthesize information from multiple sources when applicable
                7. If the sources don't contain relevant information, explain what's missing
                8. Use proper formatting to organize information clearly
                """},
                {"role": "user", "content": f"""
                Sources:
                {formatted_contexts}

                Question:
                {query}

                Please provide a comprehensive answer with source citations, including detailed explanations and examples where appropriate.
                """}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Slightly increased for more natural responses while maintaining consistency
                max_tokens=1000,  # Increased to allow for more detailed responses
                seed=42  # Fixed seed for consistent sampling
            )

            response_text = response.choices[0].message.content.strip()
            # Cache the response
            self._response_cache[cache_key] = response_text
            return response_text

        except Exception as e:
            raise Exception(f"Error generating response with sources: {str(e)}")
