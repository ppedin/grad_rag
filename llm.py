import llm_keys
from openai import OpenAI
from typing import Dict, Any, Optional

# Initialize OpenAI client
client = OpenAI(api_key=llm_keys.OPENAI_KEY)


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
