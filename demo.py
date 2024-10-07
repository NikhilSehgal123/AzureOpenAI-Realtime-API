from agent import RealtimeAPIAgent
from typing import Dict, Any
from data_models import Tool
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
AZURE_OPENAI_API_URL = os.getenv("AZURE_OPENAI_API_URL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Session instructions
INSTRUCTIONS = "You are a helpful assistant for a company called Helfie, a company that provides a platform for healthcare providers to manage their patients and appointments. You are a part of the customer support team and you are tasked with helping customers with their queries and issues. You are also tasked with sending OTPs to customers for authentication purposes."

def send_otp(phone_number: str) -> Dict[str, Any]:
    url = "https://api.prod.helfie.ai/iam/api/auth/send_otp"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "to": phone_number
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return {"message": "OTP sent successfully", "status": response.status_code}
    except requests.exceptions.RequestException as e:
        return {"message": f"Error sending OTP: {str(e)}", "status": getattr(e.response, 'status_code', None)}

# Tools configuration
TOOLS = [
    Tool(
        definition={
            "type": "function",
            "name": "send_otp",
            "description": "Send an OTP to a phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": { "type": "string" }
                },
                "required": ["phone_number"]
            }
        },
        callable=send_otp
    )
]

# Initialize the RealtimeAPIAgent with the configured URL, API key, instructions, and tools
realtime_api = RealtimeAPIAgent(
    url=AZURE_OPENAI_API_URL,
    api_key=AZURE_OPENAI_API_KEY,
    instructions=INSTRUCTIONS,
    tools=TOOLS
)

# Connect to the WebSocket server
realtime_api.connect()