from typing import List
import httpx
from openai import OpenAI
from config.settings import OPENAI_API_KEY

class Chatbot:
    def __init__(self):
        """Initialize the chatbot with OpenAI API key."""
        http_client = httpx.Client()
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=http_client
        )
        self.model = "gpt-3.5-turbo"  # Using the more modern chat model

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
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"""
                Context:
                {context}

                Question:
                {query}

                Please provide a clear and concise answer based on the context above.
                """}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def generate_response_with_sources(self, contexts: List[dict], query: str) -> tuple:
        """
        Generate a response with source citations using OpenAI's API.
        
        Args:
            contexts (List[dict]): List of context dictionaries with text and metadata
            query (str): User's query
            
        Returns:
            tuple: (response text, list of sources used)
            
        Raises:
            Exception: If there's an error in generating the response
        """
        try:
            # Format contexts with source information
            formatted_contexts = "\n\n".join([
                f"Source {i+1}:\n{ctx['text']}"
                for i, ctx in enumerate(contexts)
            ])

            messages = [
                {"role": "system", "content": """
                You are a helpful assistant that answers questions based on provided sources.
                For each response:
                1. Use information only from the provided sources
                2. Cite sources using [Source X] notation
                3. If the sources don't contain relevant information, say so
                """},
                {"role": "user", "content": f"""
                Sources:
                {formatted_contexts}

                Question:
                {query}

                Please provide a clear answer with source citations.
                """}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"Error generating response with sources: {str(e)}")
