import websocket
import json
import logging
import pyaudio
import base64
import uuid
from termcolor import colored
import threading
import time
import queue
from typing import List, Dict, Any, Callable, Optional, Union
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from data_models import Tool

class RealtimeAPIAgent:
    def __init__(self, url: str, api_key: str, instructions: str, tools: List[Tool], use_aec: bool = False):
        # WebSocket and API configuration
        self.url: str = url
        self.openai_api_key: str = api_key
        self.instructions: str = instructions
        self.tools: List[Dict[str, Any]] = [tool.definition for tool in tools]
        self.tool_callables: Dict[str, Callable] = {tool.definition['name']: tool.callable for tool in tools}
        self.ws: Optional[websocket.WebSocketApp] = None
        self.streaming: bool = False
        self.current_response_id: Optional[str] = None

        # Audio configuration
        self.p: pyaudio.PyAudio = pyaudio.PyAudio()
        self.CHUNK_SIZE: int = 1024
        self.FORMAT: int = pyaudio.paInt16
        self.CHANNELS: int = 1
        self.RATE: int = 24000
        self.output_stream: Optional[pyaudio.Stream] = None
        self.input_stream: Optional[pyaudio.Stream] = None

        # Threading and synchronization
        self.audio_thread: Optional[threading.Thread] = None
        self.playback_thread: Optional[threading.Thread] = None
        self.audio_lock: threading.Lock = threading.Lock()
        self.stream_active: threading.Event = threading.Event()
        self.audio_output_queue: queue.Queue = queue.Queue()

        # State flags
        self.is_ai_speaking: bool = False

        # Add new attributes for audio processing
        self.is_user_speaking: bool = False
        self.vad_threshold: float = 0.05  # Lowered from 0.1
        
        # Add AEC flag
        self.use_aec: bool = use_aec
        
        # AEC parameters (only initialize if AEC is enabled)
        if self.use_aec:
            self.filter_length = 1024  # Length of adaptive filter
            self.adaptive_filter = np.zeros(self.filter_length)
            self.reference_buffer = np.zeros(self.filter_length)
            self.mu = 0.1  # Step size for adaptive filter
            self.eps = 1e-6  # Small value to prevent division by zero

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Add a debug flag
        self.debug: bool = True

        # Set up rich console for prettier logging
        self.console = Console()

    def log_event(self, direction: str, event: Dict[str, Any]) -> None:
        """Log WebSocket events with rich formatting."""
        event_type = event.get('type', 'Unknown')
        if 'delta' not in event_type.lower():
            color = "cyan" if direction == "Sent" else "yellow"
            json_str = json.dumps(event, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            panel = Panel(
                syntax,
                title=f"[{color}]{direction}: {event_type}[/{color}]",
                expand=False,
                border_style=color
            )
            self.console.print(panel)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection opening."""
        self.console.print("[bold green]Connected to server.[/bold green]")
        self._send_session_update()
        self._start_audio_streaming()

    def _send_session_update(self) -> None:
        """Send initial session update to the server."""
        session_update: Dict[str, Any] = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.instructions,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "tools": self.tools,
                "tool_choice": "auto",
                "temperature": 0.6
            }
        }
        self.ws.send(json.dumps(session_update))

    def _start_audio_streaming(self) -> None:
        """Start the audio streaming thread."""
        self.audio_thread = threading.Thread(target=self.stream_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data: Dict[str, Any] = json.loads(message)
            self.log_event("Received", data)

            # This dictionary maps the websocket server event types to the handler functions
            handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {
                'response.created': self._handle_response_created,                      # This is the first event that is sent
                'input_audio_buffer.speech_started': self._handle_speech_started,       # Server has detected the users speech
                'response.audio.delta': self._handle_audio_delta,                       # Server is returning audio packets as the AI speaks
                'response.audio.done': self._handle_audio_done,                         # Server has completed sending all audio packets
                'response.audio_transcript.done': self._handle_transcript_done,         # Server has completed the audio transcript
                'response.done': self._handle_response_done,                            # Server has completed the response
                'error': self._handle_error,                                            # Server has encountered an error
            }

            handler = handlers.get(data['type'])
            if handler:
                handler(data)
            else:
                self.logger.info(f"Unhandled message type: {data['type']}")
        except KeyError as e:
            self.logger.error(f"KeyError in on_message: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in on_message: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in on_message: {e}")

    def _handle_response_created(self, data: Dict[str, Any]) -> None:
        """Handle response creation event."""
        self.current_response_id = data['response']['id']
        self.is_ai_speaking = True
        self.start_playback_thread()

    def _handle_speech_started(self, data: Dict[str, Any]) -> None:
        """Handle user speech start event."""
        self.console.print("[bold magenta]User started speaking...[/bold magenta]")
        self.console.print(f"[magenta]AI is speaking: {self.is_ai_speaking}[/magenta]")
        self.is_user_speaking = True
        # Stop AI speech
        self.cancel_response()

    def _handle_audio_delta(self, data: Dict[str, Any]) -> None:
        """Handle incoming audio data."""
        if self.is_ai_speaking:
            audio_chunk: bytes = base64.b64decode(data['delta'])
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Update the reference buffer with AI speech only if AEC is enabled
            if self.use_aec:
                if len(self.reference_buffer) < len(audio_array):
                    self.reference_buffer = np.pad(self.reference_buffer, (0, len(audio_array) - len(self.reference_buffer)))
                self.reference_buffer = np.roll(self.reference_buffer, -len(audio_array))
                self.reference_buffer[-len(audio_array):] = audio_array
            
            if self.is_user_speaking:
                # Apply audio ducking
                audio_array = (audio_array * 0.5).astype(np.int16)
            
            self.audio_output_queue.put(audio_array.tobytes())

    def _handle_audio_done(self, data: Dict[str, Any]) -> None:
        """Handle end of audio stream."""
        self.audio_output_queue.put(None)
        self.is_ai_speaking = False
        self.is_user_speaking = False  # Reset user speaking flag

    def _handle_transcript_done(self, data: Dict[str, Any]) -> None:
        """Handle completed transcript."""
        self.console.print(f"[bold green]Transcript:[/bold green] {data['transcript']}")

    def _handle_response_done(self, data: Dict[str, Any]) -> None:
        """Handle completed response."""
        self.current_response_id = None
        self.is_ai_speaking = False
        function_call_item = next((item for item in data['response']['output'] if item['type'] == 'function_call'), None)
        
        if function_call_item:
            self._send_function_call_output(function_call_item)

    def _send_function_call_output(self, function_call_item: Dict[str, Any]) -> None:
        """Execute the function call and send its output."""
        function_name: str = function_call_item['name']
        function_args: Dict[str, Any] = json.loads(function_call_item['arguments'])
        
        if function_name in self.tool_callables:
            try:
                result: Dict[str, Any] = self.tool_callables[function_name](**function_args)
                output: str = json.dumps(result)
                status: str = "completed"
            except Exception as e:
                output: str = json.dumps({"error": str(e)})
                status: str = "error"
        else:
            output: str = json.dumps({"error": f"Function {function_name} not found"})
            status: str = "error"
        
        function_call_output_item: Dict[str, Any] = {
            "id": f"item_{uuid.uuid4().hex[:24]}",
            "type": "function_call_output",
            "status": status,
            "output": output,
            "call_id": function_call_item['call_id']
        }
        
        conversation_item_create_event: Dict[str, Any] = {
            "type": "conversation.item.create",
            "event_id": f"event_{uuid.uuid4().hex[:24]}",
            "previous_item_id": function_call_item['id'],
            "item": function_call_output_item
        }

        self.log_event("Sent", conversation_item_create_event)
        self.ws.send(json.dumps(conversation_item_create_event))

        # Now trigger a response by sending a response.create event
        response_create_event: Dict[str, Any] = {
            "type": "response.create"
        }

        self.log_event("Sent", response_create_event)
        self.ws.send(json.dumps(response_create_event))

    def _handle_error(self, data: Dict[str, Any]) -> None:
        """Handle error events."""
        error_message = json.dumps(data, indent=2)
        panel = Panel(
            Syntax(error_message, "json", theme="monokai", line_numbers=True),
            title="[bold red]Error Received[/bold red]",
            expand=False,
            border_style="red"
        )
        self.console.print(panel)

    def clear_input_audio_buffer(self) -> None:
        clear_buffer_event: Dict[str, Any] = {
            "type": "input_audio_buffer.clear",
            "event_id": f"event_{uuid.uuid4().hex[:24]}"
        }
        self.log_event("Sent", clear_buffer_event)
        self.ws.send(json.dumps(clear_buffer_event))

    def start_playback_thread(self) -> None:
        self.stop_playback()  # Ensure any existing playback is stopped
        self.playback_thread = threading.Thread(target=self.playback_audio)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        self.stream_active.set()  # Ensure the stream is marked as active

    def playback_audio(self) -> None:
        try:
            with self.audio_lock:
                self.output_stream = self.p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    output=True
                )
                self.stream_active.set()
            
            while self.stream_active.is_set():
                try:
                    chunk: Optional[bytes] = self.audio_output_queue.get(timeout=1)
                    if chunk is None:
                        break
                    with self.audio_lock:
                        if self.stream_active.is_set():
                            self.output_stream.write(chunk)
                except queue.Empty:
                    continue
                except OSError as e:
                    self.logger.error(f"PortAudio error: {e}")
                    break
        except Exception as e:
            self.logger.error(f"Error in playback_audio: {e}")
        finally:
            self.stop_playback()

    def stop_playback(self) -> None:
        with self.audio_lock:
            self.stream_active.clear()
            if self.output_stream:
                try:
                    if self.output_stream.is_active():
                        self.output_stream.stop_stream()
                    self.output_stream.close()
                except Exception as e:
                    self.logger.error(f"Error stopping playback stream: {e}")
                finally:
                    self.output_stream = None
        
        self.logger.info(colored("Playback stopped.", "red"))
        
        # Clear any remaining audio in the queue
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
            except queue.Empty:
                break

    def stream_audio(self) -> None:
        self.input_stream = self.p.open(format=self.FORMAT,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              frames_per_buffer=self.CHUNK_SIZE)
        
        self.logger.info(colored("Audio streaming started.", "green"))
        
        while self.streaming:
            try:
                audio_data: bytes = self.input_stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Apply echo cancellation if enabled
                if self.use_aec:
                    audio_array = self.apply_echo_cancellation(audio_array)
                
                # Perform voice activity detection
                is_speech = self.is_voice_activity(audio_array)
                
                if self.debug:
                    self.logger.debug(f"Voice activity detected: {is_speech}")
                
                if is_speech:
                    self.is_user_speaking = True
                    base64_audio: str = base64.b64encode(audio_array.tobytes()).decode('utf-8')
                    
                    audio_event: Dict[str, Any] = {
                        "type": "input_audio_buffer.append",
                        "event_id": f"event_{uuid.uuid4().hex[:24]}",
                        "audio": base64_audio
                    }
                    
                    if self.ws and self.ws.sock and self.ws.sock.connected:
                        self.ws.send(json.dumps(audio_event))
                        if self.debug:
                            self.logger.debug("Sent audio data to server")
                    else:
                        self.logger.info(colored("WebSocket not connected. Stopping audio streaming.", "red"))
                        break
                else:
                    self.is_user_speaking = False

            except websocket.WebSocketConnectionClosedException:
                self.logger.info(colored("WebSocket connection closed. Stopping audio streaming.", "red"))
                break
            except Exception as e:
                self.logger.error(colored(f"Error in audio streaming: {e}", "red"))
                break
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        self.logger.info(colored("Audio streaming stopped.", "red"))

    def connect(self) -> None:
        """Establish and maintain WebSocket connection."""
        headers: Dict[str, str] = {
            "api-key": self.openai_api_key,
        }

        self.streaming = True
        while self.streaming:
            try:
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_close=self.on_close,
                    on_error=self.on_error,
                    header=headers
                )
                
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except KeyboardInterrupt:
                self.logger.info(colored("Keyboard interrupt received. Stopping...", "red"))
                self.streaming = False
                break
            except Exception as e:
                self.logger.error(colored(f"WebSocket connection error: {e}", "red"))
                time.sleep(5)  # Wait before reconnecting
            finally:
                self.close_connection()

        self.disconnect()

    def close_connection(self) -> None:
        self.streaming = False
        if self.ws:
            self.ws.close()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5)

    def on_close(self, ws: websocket.WebSocketApp, close_status_code: Optional[int], close_msg: Optional[str]) -> None:
        self.console.print(f"[bold red]WebSocket connection closed: {close_status_code} - {close_msg}[/bold red]")
        self.streaming = False

    def on_error(self, ws: websocket.WebSocketApp, error: Union[Exception, str]) -> None:
        self.console.print(f"[bold red]WebSocket error: {error}[/bold red]")
        self.streaming = False

    def disconnect(self) -> None:
        self.console.print("[bold red]Disconnecting...[/bold red]")
        self.streaming = False
        self.close_connection()
        self.stop_playback()
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=5)
        self.p.terminate()
        self.console.print("[bold red]Disconnected and audio streaming stopped. Exiting...[/bold red]")

    def cancel_response(self) -> None:
        if self.current_response_id:
            cancel_event: Dict[str, Any] = {
                "type": "response.cancel",
                "event_id": f"event_{uuid.uuid4().hex[:24]}"
            }
            self.log_event("Sent", cancel_event)
            self.ws.send(json.dumps(cancel_event))
    
        self.is_ai_speaking = False
        self.current_response_id = None
        self.stop_playback()  # This will stop the audio immediately
        self.audio_output_queue.queue.clear()  # Clear any pending audio

    def _handle_output_item_done(self, data: Dict[str, Any]) -> None:
        """Handle output item done event."""
        item = data.get('item', {})
        if item.get('type') == 'function_call':
            self._handle_function_call(item)

    def _handle_function_call(self, item: Dict[str, Any]) -> None:
        """Handle function call item."""
        function_name = item.get('name')
        arguments = item.get('arguments')
        if function_name and arguments:
            try:
                self._send_function_call_output(item)
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON in function arguments: {arguments}")

    def _handle_function_call_arguments_done(self, data: Dict[str, Any]) -> None:
        """Handle function call arguments done event."""
        item_id = data.get('item_id')
        name = data.get('name')
        arguments = data.get('arguments')
        if item_id and name and arguments:
            try:
                self._send_function_call_output({
                    'id': item_id,
                    'name': name,
                    'arguments': arguments
                })
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON in function arguments: {arguments}")

    def apply_echo_cancellation(self, audio_array: np.ndarray) -> np.ndarray:
        if not self.use_aec:
            return audio_array

        """Apply Normalized Least Mean Squares (NLMS) adaptive filtering for echo cancellation."""
        input_length = len(audio_array)
        
        # Pad or truncate the reference buffer to match the input length
        if len(self.reference_buffer) < input_length:
            self.reference_buffer = np.pad(self.reference_buffer, (0, input_length - len(self.reference_buffer)))
        elif len(self.reference_buffer) > input_length:
            self.reference_buffer = self.reference_buffer[-input_length:]
        
        # Pad or truncate the adaptive filter if necessary
        if len(self.adaptive_filter) < input_length:
            self.adaptive_filter = np.pad(self.adaptive_filter, (0, input_length - len(self.adaptive_filter)))
        elif len(self.adaptive_filter) > input_length:
            self.adaptive_filter = self.adaptive_filter[:input_length]
        
        # Estimate echo using adaptive filter
        echo_estimate = np.convolve(self.reference_buffer, self.adaptive_filter, mode='same')

        # Subtract echo estimate from input
        error = audio_array - echo_estimate

        # Update adaptive filter
        if self.is_ai_speaking:
            x = self.reference_buffer
            update = self.mu * error / (np.dot(x, x) + self.eps)
            self.adaptive_filter += update * x

        # Update reference buffer with new AI audio
        self.reference_buffer = np.roll(self.reference_buffer, -input_length)
        self.reference_buffer[-input_length:] = audio_array

        if self.debug:
            self.logger.debug(f"Echo cancellation applied. Input energy: {np.sum(np.abs(audio_array))}, Output energy: {np.sum(np.abs(error))}")

        return error.astype(np.int16)

    def lower_ai_volume(self) -> None:
        """Lower the volume of AI speech."""
        with self.audio_lock:
            if self.output_stream:
                current_volume = self.output_stream.get_volume()
                self.output_stream.set_volume(current_volume * 0.5)  # Reduce volume by 50%

    def is_voice_activity(self, audio_array: np.ndarray) -> bool:
        """Simple energy-based voice activity detection."""
        energy = np.sum(np.abs(audio_array)) / len(audio_array)
        is_speech = energy > self.vad_threshold
        if self.debug:
            self.logger.debug(f"Voice activity detection: Energy = {energy}, Threshold = {self.vad_threshold}, Is speech = {is_speech}")
        return is_speech

    def debug_audio_input(self):
        """Debug method to check raw audio input."""
        print("Starting audio input debug. Press Ctrl+C to stop.")
        try:
            while True:
                audio_data = self.input_stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                energy = np.sum(np.abs(audio_array)) / len(audio_array)
                print(f"Raw audio energy: {energy}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Audio input debug stopped.")