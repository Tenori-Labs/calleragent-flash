#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys
from typing import AsyncGenerator

# Monkeypatch for HTTPMethod import compatibility
if sys.version_info < (3, 11):
    import http
    if not hasattr(http, 'HTTPMethod'):
        from enum import Enum
        class HTTPMethod(Enum):
            GET = "GET"
            POST = "POST"
            PUT = "PUT"
            DELETE = "DELETE"
            PATCH = "PATCH"
            HEAD = "HEAD"
            OPTIONS = "OPTIONS"
        http.HTTPMethod = HTTPMethod

from dotenv import load_dotenv
from loguru import logger
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI, WebSocket
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.ultravox.stt import UltravoxSTTService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

# NOTE: This bot requires GPU resources to run efficiently.
# The Ultravox model is compute-intensive and performs best with GPU acceleration.
# Deploy on cloud GPU providers like Cerebrium.ai for optimal performance.

# Performance optimization environment variables
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "1"  # Enable Triton Flash Attention
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # Use Flash Attention backend
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better performance
os.environ["TORCH_USE_CUDA_DSA"] = "0"  # Disable for better performance
os.environ["VLLM_USE_V2_BLOCK_MANAGER"] = "1"  # Use v2 block manager for better memory
os.environ["VLLM_USE_RAY_SCHEDULER"] = "1"  # Use Ray scheduler for better performance

# Setup ngrok proxy
def setup_ngrok_proxy():
    """Setup ngrok tunnel for telephony transport."""
    ngrok_auth_token = os.getenv("NGROK_AUTHTOKEN")
    if not ngrok_auth_token:
        logger.error("NGROK_AUTHTOKEN not found in environment variables")
        raise ValueError("NGROK_AUTHTOKEN is required for telephony transport")
    
    # Set ngrok auth token
    ngrok.set_auth_token(ngrok_auth_token)
    
    # Create HTTP tunnel
    tunnel = ngrok.connect(7860, "http")
    proxy_url = tunnel.public_url
    
    logger.info(f"Ngrok tunnel created: {proxy_url}")
    return proxy_url


class UltravoxLLMService(UltravoxSTTService):
    """Extended Ultravox service that supports system prompts and acts as a full LLM."""
    
    def __init__(self, system_prompt: str = None, max_conversation_turns: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._conversation_messages = []  # Store text-based conversation history
        self._current_user_transcription = ""  # Store the current user's speech as text
        self._max_conversation_turns = max_conversation_turns
        self._error_count = 0
        logger.info(f"Initialized UltravoxLLMService with system prompt and conversation memory")
    
    def _manage_conversation_history(self):
        """Keep conversation history within limits to prevent memory issues."""
        # Keep only the last N turns (user + assistant pairs)
        max_messages = self._max_conversation_turns * 2  # Each turn has user + assistant
        if len(self._conversation_messages) > max_messages:
            # Remove oldest messages, keeping the most recent ones
            self._conversation_messages = self._conversation_messages[-max_messages:]
            logger.info(f"Trimmed conversation history to {len(self._conversation_messages)} messages")
    
    async def _process_audio_buffer(self) -> AsyncGenerator[Frame, None]:
        """Process collected audio frames with Ultravox using system prompt."""
        try:
            self._buffer.is_processing = True

            if not self._buffer.frames:
                logger.warning("No audio frames to process")
                return

            # Process audio frames
            audio_arrays = []
            for f in self._buffer.frames:
                if hasattr(f, "audio") and f.audio:
                    if isinstance(f.audio, bytes):
                        try:
                            import numpy as np
                            arr = np.frombuffer(f.audio, dtype=np.int16)
                            if arr.size > 0:
                                audio_arrays.append(arr)
                        except Exception as e:
                            logger.error(f"Error processing bytes audio frame: {e}")
                    elif isinstance(f.audio, np.ndarray):
                        if f.audio.size > 0:
                            if f.audio.dtype != np.int16:
                                audio_arrays.append(f.audio.astype(np.int16))
                            else:
                                audio_arrays.append(f.audio)

            if not audio_arrays:
                logger.warning("No valid audio data found in frames")
                return

            # Concatenate and convert audio
            import numpy as np
            audio_data = np.concatenate(audio_arrays)
            audio_int16 = audio_data
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Validate and clean audio data
            if len(audio_float32) == 0:
                logger.warning("Empty audio data, skipping processing")
                return
            
            # Ensure audio is within reasonable bounds
            if np.any(np.isnan(audio_float32)) or np.any(np.isinf(audio_float32)):
                logger.warning("Invalid audio data (NaN/Inf), skipping processing")
                return
            
            # Limit audio length to prevent memory issues and CUDA errors
            max_audio_length = int(16000 * 5)  # 5 seconds max
            if len(audio_float32) > max_audio_length:
                audio_float32 = audio_float32[:max_audio_length]
                logger.info(f"Truncated audio to {max_audio_length} samples")
            
            # Ensure audio is not too short (at least 0.5 seconds)
            min_audio_length = int(16000 * 0.5)  # 0.5 seconds min
            if len(audio_float32) < min_audio_length:
                # Pad with silence
                padding_length = int(min_audio_length - len(audio_float32))
                padding = np.zeros(padding_length, dtype=np.float32)
                audio_float32 = np.concatenate([audio_float32, padding])
                logger.info(f"Padded audio to {len(audio_float32)} samples")

            # Build messages for Ultravox model
            # FIXED: Build conversation context into the system prompt instead of adding as separate message
            system_prompt_with_context = self._system_prompt
            
            if self._conversation_messages:
                # Add conversation history to system prompt
                context = "\n\nPrevious conversation:\n"
                for msg in self._conversation_messages:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    context += f"{role}: {content}\n"
                
                system_prompt_with_context += context
                logger.info(f"Added conversation context with {len(self._conversation_messages)} previous messages to system prompt")
            
            messages_for_model = [
                {"role": "system", "content": system_prompt_with_context},
                {"role": "user", "content": "<|audio|>\n"}
            ]

            logger.info(f"Generating response with {len(messages_for_model)} messages in context")

            # Generate text using the model
            if self._model:
                try:
                    await self.start_ttfb_metrics()
                    await self.start_processing_metrics()

                    from pipecat.frames.frames import (
                        LLMFullResponseStartFrame,
                        LLMTextFrame,
                        LLMFullResponseEndFrame,
                    )
                    import json

                    yield LLMFullResponseStartFrame()

                    full_response = ""
                    user_speech = ""  # Will extract the user's speech from the response
                    first_chunk_received = False
                    
                    # Add debugging information
                    logger.info(f"Audio shape: {audio_float32.shape}, dtype: {audio_float32.dtype}")
                    logger.info(f"Messages count: {len(messages_for_model)}")
                    
                    # Ensure audio is in the correct format
                    if len(audio_float32) == 0:
                        logger.warning("Empty audio data, using placeholder")
                        audio_float32 = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                    
                    # Limit audio length to prevent memory issues
                    max_audio_length = int(16000 * 10)  # 10 seconds max
                    if len(audio_float32) > max_audio_length:
                        audio_float32 = audio_float32[:max_audio_length]
                        logger.info(f"Truncated audio to {max_audio_length} samples")
                    
                    async for response in self._model.generate(
                        messages=messages_for_model,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        audio=audio_float32,
                    ):
                        await self.stop_ttfb_metrics()

                        chunk = json.loads(response)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                new_text = delta["content"]
                                if new_text:
                                    full_response += new_text
                                    yield LLMTextFrame(text=new_text)

                    await self.stop_processing_metrics()
                    yield LLMFullResponseEndFrame()

                    # Store the conversation in history as TEXT (not audio placeholders)
                    # The model generates responses that implicitly understand what was said
                    # We'll store a placeholder for user input and the actual response
                    
                    # Add user message (we don't have transcription, but we can infer from response)
                    # For now, use a placeholder that indicates audio was provided
                    if full_response:
                        # Only store conversation if we got a valid response
                        user_placeholder = "[User spoke via audio]"
                        self._conversation_messages.append({"role": "user", "content": user_placeholder})
                        self._conversation_messages.append({"role": "assistant", "content": full_response})
                        
                        logger.info(f"Stored conversation turn in history (total: {len(self._conversation_messages)} messages)")
                    
                    # Reset error count on successful generation
                    self._error_count = 0
                    
                    # Manage conversation history size
                    self._manage_conversation_history()

                    logger.info(f"Generated response: {full_response[:100]}...")

                except Exception as e:
                    logger.error(f"Error generating text from model: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    
                    # Increment error count
                    self._error_count += 1
                    
                    # If too many errors, reset conversation history
                    if self._error_count >= 3:
                        logger.warning("Too many errors, resetting conversation history")
                        self._conversation_messages = []
                        self._error_count = 0
                    
                    # Try to provide a fallback response
                    fallback_response = "I'm sorry, I encountered a technical issue. Could you please try again?"
                    from pipecat.frames.frames import ErrorFrame, LLMTextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
                    
                    yield LLMFullResponseStartFrame()
                    yield LLMTextFrame(text=fallback_response)
                    yield LLMFullResponseEndFrame()
                    
                    # Don't store this as conversation history since it's an error
            else:
                logger.warning("No model available for text generation")
                from pipecat.frames.frames import ErrorFrame
                yield ErrorFrame("No model available for text generation")

        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            from pipecat.frames.frames import ErrorFrame
            yield ErrorFrame(f"Error processing audio: {str(e)}")
        finally:
            self._buffer.is_processing = False
            self._buffer.frames = []
            self._buffer.started_at = None


# Initialize Ultravox LLM processor with system prompt and performance optimizations
ultravox_llm = UltravoxLLMService(
    model_name="fixie-ai/ultravox-v0_6-llama-3_3-70b",
    hf_token=os.getenv("HF_TOKEN"),
    system_prompt=(
        "MOST IMPORTANT: Talk in Colloquial Tamil with a mixture of Tamil and English words.\n"
        "Use TAMIL literals for generating Tamil words and English literals for English words.\n\n"

        "You are a helpful AI assistant in a phone call. Your goal is to demonstrate "
        "your capabilities in a succinct way. Keep your responses concise and natural "
        "for voice conversation. Don't include special characters in your answers. "
        "Respond to what the user said in a creative and helpful way.\n\n"

        "IMPORTANT - CONVERSATION MEMORY:\n"
        "- You will see 'Previous conversation:' sections showing what was said before.\n"
        "- ALWAYS read and remember ALL information from previous exchanges.\n"
        "- Use this context to provide relevant, connected responses.\n"
        "- If the user asks about something mentioned earlier, recall it from the conversation history.\n"
        "- Reference previous topics naturally to show you remember the conversation.\n"
        "- Treat the entire call as ONE continuous conversation, not separate interactions.\n\n"

        "ADDITIONAL INSTRUCTIONS (COLLOQUIAL TAMIL MODE):\n"
        "- Speak in a mix of Tamil and English words (Tanglish) in a friendly, casual tone.\n"
        "- Sound like a native Tamil speaker chatting informally — natural and expressive.\n"
        "- Use light humor, friendly fillers, and casual phrasing.\n"
        "- Keep sentences short and conversational, as if talking over a phone call.\n"
        "- Avoid being overly formal or robotic; sound warm and human-like.\n"
        "- If explaining something complex, mix Tamil and English naturally.\n\n"

        "EXAMPLES OF HOW TO SPEAK (TANGLISH STYLE):\n\n"
        "Example 1:\n"
        "User: Hey, what are you doing?\n"
        "Assistant: சும்மா தான் டா, coffee குடிக்கறேன். நீ என்ன பண்ணுறே?\n\n"

        "Example 2:\n"
        "User: Can you explain what AI means?\n"
        "Assistant: ஆமா ஆமா, AIன்னா Artificial Intelligence டா — basically, machine நம்ம மாதிரி think பண்ணும், learn பண்ணும்.\n\n"

        "Example 3:\n"
        "User: Weather எப்படி இருக்கு அங்கே?\n"
        "Assistant: இங்க நாறா சூடா இருக்கு டா, fan full speedல போடணும் போல இருக்கு!\n\n"

        "Example 4:\n"
        "User: Tell me a joke.\n"
        "Assistant: சரி, ஓன்னு கேள் — ஒரு computerக்கு fever வந்தா, அது சொல்லும் I've got a virus! ஹா ஹா!\n\n"

        "Example 5:\n"
        "User: Can you help me with my project?\n"
        "Assistant: ஆமா டா, சொல்லு என்ன project. நம்ம சேர்ந்து பண்ணலாம் easyஆ.\n\n"

        "Remember: Mix Tamil and English naturally, keep it friendly and human, like a real phone chat between buddies."

        "MOST VERY VERY IMPORTANT: The TAMIL should be MORE in your response THAN ENGLISH!!!!"
    ),
    temperature=0.3,
    max_tokens=100,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="bfloat16",
    enforce_eager=False,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
)

async def run_bot(transport: BaseTransport, handle_sigint: bool):
    # Initialize ElevenLabs TTS service
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="C2RGMrNBTZaNfddRPeRH",
    )

    # Create pipeline with Ultravox as the central multimodal LLM
    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from Exotel
            ultravox_llm,  # Ultravox processes audio and generates intelligent responses
            tts,  # Text-To-Speech (ElevenLabs)
            transport.output(),  # Websocket output to Exotel
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,  # Exotel telephony sample rate
            audio_out_sample_rate=16000,  # Exotel telephony sample rate
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected to Exotel bot with Ultravox LLM")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    serializer = ExotelFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            serializer=serializer,
        ),
    )

    handle_sigint = runner_args.handle_sigint

    await run_bot(transport, handle_sigint)


if __name__ == "__main__":
    import sys
    import uvicorn
    from fastapi import FastAPI, WebSocket
    from pipecat.runner.types import WebSocketRunnerArguments
    
    # Setup ngrok proxy
    try:
        proxy_url = setup_ngrok_proxy()
        logger.info(f"Ngrok proxy URL: {proxy_url}")
        logger.info(f"WebSocket URL for Exotel: {proxy_url.replace('http', 'ws')}/ws")
    except Exception as e:
        logger.error(f"Failed to setup ngrok proxy: {e}")
        sys.exit(1)

    # Create FastAPI app
    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle WebSocket connection from Exotel Media Streams."""
        await websocket.accept()
        logger.info("WebSocket connection accepted from Exotel")

        try:
            # Create runner arguments and run the bot
            runner_args = WebSocketRunnerArguments(websocket=websocket)
            runner_args.handle_sigint = False

            await bot(runner_args)

        except Exception as e:
            logger.error(f"Error in WebSocket endpoint: {e}")
            await websocket.close()

    # Start the server
    logger.info("Starting server on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860)
