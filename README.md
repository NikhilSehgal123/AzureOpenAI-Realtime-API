# OpenAI Realtime API Client (Un)Official

This project is a client for the OpenAI Realtime API. It is not official and is not affiliated with OpenAI.

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/NikhilSehgal123/AzureOpenAI-Realtime-API.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the root directory of the project.

2. Add the following environment variables to the `.env` file:
   ```
   AZURE_OPENAI_API_URL="your_azure_openai_websocket_url"
   AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
   ```
   Replace `your_azure_openai_websocket_url` and `your_azure_openai_api_key` with your actual Azure OpenAI API WebSocket URL and API key.

## Usage

To run the AI assistant:

1. Ensure you're in the project directory.

2. Run the demo script:
   ```
   python demo.py
   ```

3. The assistant will connect to the Azure OpenAI API and start listening for WebSocket events.

## Features

- Real-time text and audio communication
- Voice activity detection
- Echo cancellation (optional)
- Function calling (e.g., sending OTPs)
- Rich console output for easy debugging

## Customization

You can customize the assistant's behavior by modifying the `INSTRUCTIONS` and `TOOLS` in the `demo.py` file. Add new tools or change the instructions to fit your specific use case.

## Troubleshooting

If you encounter any issues:

1. Ensure your `.env` file is correctly set up with valid Azure OpenAI credentials.
2. Check your internet connection.
3. Verify that you have the required Python version and all dependencies installed.

For more detailed logs, set the `debug` flag to `True` in the `RealtimeAPIAgent` initialization.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.