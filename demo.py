from agent import RealtimeAPIAgent
from typing import Dict, Any
from data_models import Tool
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Tools
def get_weather(city: str) -> Dict[str, Any]:
    # Return a dummy response for now
    return {"weather": "sunny", "temperature": "75 degrees fahrenheit"}

# Tools configuration
TOOLS = [
    Tool(
        definition={
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }
        },
        callable=get_weather
    )
]

# Initialize the RealtimeAPIAgent with the configured URL, API key, instructions, and tools
realtime_api = RealtimeAPIAgent(
    url=os.getenv("AZURE_OPENAI_API_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    instructions="You are a helpful assistant.",
    tools=TOOLS
)

# Connect to the WebSocket server
realtime_api.connect()