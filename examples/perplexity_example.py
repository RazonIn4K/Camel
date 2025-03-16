"""
Example script demonstrating how to use the Perplexity API with Gray Swan Arena.

This script shows how to load configuration and use the Perplexity API
with the OpenAI client library (which Perplexity is compatible with).
"""

import os
from pathlib import Path
from openai import OpenAI
from camel.types import ModelType, ModelPlatformType

# Import Gray Swan Arena config
from cybersec_agents.grayswan.config import Config

# Load configuration
config = Config(config_dir=Path("config"))

# Get Perplexity model configuration
perplexity_config = config.get_model_config(
    model_type=ModelType.SONA_PRO, 
    model_platform=ModelPlatformType.PERPLEXITY
)

# Get API key from environment variable
api_key = os.getenv(perplexity_config.api_key_env)
if not api_key:
    raise ValueError(f"API key not found in environment variable: {perplexity_config.api_key_env}")

# Create OpenAI client with Perplexity configuration
client = OpenAI(
    api_key=api_key,
    base_url=perplexity_config.base_url
)

# Set up the system and user messages
messages = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant specialized in cybersecurity. "
            "Provide detailed but concise responses focused on cybersecurity topics."
        ),
    },
    {
        "role": "user",
        "content": "What are the main differences between symmetric and asymmetric encryption?"
    }
]

# Make the API call
try:
    response = client.chat.completions.create(
        model="sonar-small-online",  # Perplexity model name
        messages=messages,
        temperature=perplexity_config.temperature,
        max_tokens=perplexity_config.max_tokens
    )
    
    # Print the response
    print("Response from Perplexity API:")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"Error calling Perplexity API: {e}")
    
print("\n\nNote: To use this example, set the PERPLEXITY_API_KEY environment variable:")
print("export PERPLEXITY_API_KEY='your_perplexity_api_key'") 