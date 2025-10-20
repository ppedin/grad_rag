import llm_keys
from openai import OpenAI
from mistralai import Mistral
from typing import Dict, Any, Optional, List

# Initialize OpenAI client
client = OpenAI(api_key=llm_keys.OPENAI_KEY)

# Initialize Mistral client
mistral_client = Mistral(api_key=llm_keys.MISTRAL_KEY)


def call_openai_structured(system_prompt: str, response_format: Dict[str, Any], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Call OpenAI API with structured output functionality.

    Args:
        system_prompt (str): The system prompt for the LLM
        response_format (Dict[str, Any]): The JSON schema for structured output
        model (str): OpenAI model to use (default: gpt-4o-mini)

    Returns:
        Dict[str, Any]: The structured response from the LLM

    Raises:
        Exception: If the API call fails
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ""}  # Empty user prompt as specified
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": response_format
                }
            },
            temperature=0.1  # Low temperature for more consistent structured output
        )

        # Parse the JSON response
        import json
        structured_response = json.loads(response.choices[0].message.content)

        return {
            "status": "success",
            "response": structured_response,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "response": None,
            "usage": None
        }


def get_embeddings(text_chunks: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Get embeddings for a list of text chunks using OpenAI's embedding API.

    Args:
        text_chunks (List[str]): List of text chunks to embed
        model (str): OpenAI embedding model to use (default: text-embedding-3-small)

    Returns:
        List[List[float]]: List of embedding vectors (one for each input chunk)

    Raises:
        Exception: If the API call fails
    """
    try:
        response = client.embeddings.create(
            input=text_chunks,
            model=model
        )

        # Extract embeddings from response
        embeddings = [data.embedding for data in response.data]

        return embeddings

    except Exception as e:
        raise Exception(f"Failed to get embeddings: {str(e)}")


async def get_embeddings_async(text_chunks: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Async version of get_embeddings for concurrent batch processing.

    Args:
        text_chunks (List[str]): List of text chunks to embed
        model (str): OpenAI embedding model to use (default: text-embedding-3-small)

    Returns:
        List[List[float]]: List of embedding vectors (one for each input chunk)

    Raises:
        Exception: If the API call fails
    """
    import asyncio

    # Run the synchronous OpenAI call in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    def _sync_embeddings():
        try:
            response = client.embeddings.create(
                input=text_chunks,
                model=model
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise Exception(f"Failed to get embeddings: {str(e)}")

    embeddings = await loop.run_in_executor(None, _sync_embeddings)
    return embeddings


# ===== MISTRAL EMBEDDING FUNCTIONS =====

def get_mistral_embeddings(text_chunks: List[str], model: str = "mistral-embed") -> List[List[float]]:
    """
    Get embeddings for a list of text chunks using Mistral's embedding API.

    Args:
        text_chunks (List[str]): List of text chunks to embed
        model (str): Mistral embedding model to use (default: mistral-embed)

    Returns:
        List[List[float]]: List of embedding vectors (one for each input chunk)

    Raises:
        Exception: If the API call fails
    """
    try:
        response = mistral_client.embeddings.create(
            model=model,
            inputs=text_chunks
        )

        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]

        return embeddings

    except Exception as e:
        raise Exception(f"Failed to get Mistral embeddings: {str(e)}")


async def get_mistral_embeddings_async(text_chunks: List[str], model: str = "mistral-embed") -> List[List[float]]:
    """
    Async version of get_mistral_embeddings for concurrent batch processing.

    Args:
        text_chunks (List[str]): List of text chunks to embed
        model (str): Mistral embedding model to use (default: mistral-embed)

    Returns:
        List[List[float]]: List of embedding vectors (one for each input chunk)

    Raises:
        Exception: If the API call fails
    """
    import asyncio

    # Run the synchronous Mistral call in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    def _sync_embeddings():
        try:
            response = mistral_client.embeddings.create(
                model=model,
                inputs=text_chunks
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"Failed to get Mistral embeddings: {str(e)}")

    embeddings = await loop.run_in_executor(None, _sync_embeddings)
    return embeddings
