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
import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.frames.frames import (
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    AudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.services.ultravox.stt import UltravoxSTTService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

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
    """Extended Ultravox service that supports system prompts and acts as a full LLM with conversation memory."""
    
    def __init__(self, system_prompt: str = None, max_conversation_turns: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._conversation_history = []  # Store structured conversation turns
        self._max_conversation_turns = max_conversation_turns
        self._error_count = 0
        logger.info(f"Initialized UltravoxLLMService with conversation memory (max {max_conversation_turns} turns)")
    
    def _manage_conversation_history(self):
        """Keep conversation history within limits to prevent memory issues."""
        if len(self._conversation_history) > self._max_conversation_turns:
            # Remove oldest turns, keeping the most recent ones
            removed_count = len(self._conversation_history) - self._max_conversation_turns
            self._conversation_history = self._conversation_history[-self._max_conversation_turns:]
            logger.info(f"Trimmed conversation history: removed {removed_count} old turns, kept {len(self._conversation_history)} recent turns")
    
    def _build_context_messages(self, audio_float32):
        """Build the message list for Ultravox with conversation history."""
        import numpy as np
        
        # Start with system prompt that includes conversation context
        if self._conversation_history:
            # Build a text summary of previous conversation
            context_text = "Previous conversation context:\n"
            for i, turn in enumerate(self._conversation_history, 1):
                context_text += f"\nTurn {i}:\n"
                context_text += f"User said: {turn['user']}\n"
                context_text += f"You responded: {turn['assistant']}\n"
            
            enhanced_system_prompt = f"{self._system_prompt}\n\n{context_text}\n\nNow respond to the current user's speech:"
            logger.info(f"Built context with {len(self._conversation_history)} previous turns")
        else:
            enhanced_system_prompt = self._system_prompt
            logger.info("No previous conversation history, using base system prompt")
        
        # Build messages in the format Ultravox expects
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": "<|audio|>"}  # Ultravox will replace this with audio
        ]
        
        return messages
    
    async def _process_audio_buffer(self) -> AsyncGenerator[Frame, None]:
        """Process collected audio frames with Ultravox using system prompt and conversation history."""
        try:
            self._buffer.is_processing = True

            if not self._buffer.frames:
                logger.warning("No audio frames to process")
                return

            # Process audio frames with minimal manipulation to preserve quality
            import numpy as np
            audio_arrays = []
            
            for f in self._buffer.frames:
                if hasattr(f, "audio") and f.audio:
                    if isinstance(f.audio, bytes):
                        try:
                            arr = np.frombuffer(f.audio, dtype=np.int16)
                            if arr.size > 0:
                                audio_arrays.append(arr)
                        except Exception as e:
                            logger.error(f"Error processing bytes audio frame: {e}")
                    elif isinstance(f.audio, np.ndarray):
                        if f.audio.size > 0:
                            # Ensure correct dtype
                            if f.audio.dtype != np.int16:
                                audio_arrays.append(f.audio.astype(np.int16))
                            else:
                                audio_arrays.append(f.audio)

            if not audio_arrays:
                logger.warning("No valid audio data found in frames")
                return

            # Concatenate audio
            audio_int16 = np.concatenate(audio_arrays)
            
            # Convert to float32 for the model (normalize properly)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Validate audio data
            if len(audio_float32) == 0:
                logger.warning("Empty audio data, skipping processing")
                return
            
            # Check for invalid values
            if np.any(np.isnan(audio_float32)) or np.any(np.isinf(audio_float32)):
                logger.warning("Invalid audio data (NaN/Inf), skipping processing")
                return
            
            # Clip values to valid range [-1.0, 1.0]
            audio_float32 = np.clip(audio_float32, -1.0, 1.0)
            
            # More reasonable length limits - don't cut speech too aggressively
            max_audio_length = int(16000 * 15)  # 15 seconds max (increased from 5)
            min_audio_length = int(16000 * 0.3)  # 0.3 seconds min (decreased from 0.5)
            
            if len(audio_float32) > max_audio_length:
                logger.warning(f"Audio too long ({len(audio_float32)/16000:.1f}s), truncating to {max_audio_length/16000:.1f}s")
                audio_float32 = audio_float32[:max_audio_length]
            
            if len(audio_float32) < min_audio_length:
                logger.warning(f"Audio too short ({len(audio_float32)/16000:.2f}s), padding to {min_audio_length/16000:.2f}s")
                padding_length = min_audio_length - len(audio_float32)
                padding = np.zeros(padding_length, dtype=np.float32)
                audio_float32 = np.concatenate([audio_float32, padding])

            logger.info(f"Processing audio: {len(audio_float32)/16000:.2f} seconds, {len(audio_float32)} samples")

            # Build messages with conversation context
            messages_for_model = self._build_context_messages(audio_float32)

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
                    
                    # Generate response from Ultravox
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

                    # Extract user transcription and store conversation turn
                    # Ultravox implicitly transcribes the audio and responds to it
                    # We can infer what the user said from the context, but for a proper
                    # conversation history, we should note that audio was provided
                    
                    if full_response.strip():
                        # For better conversation tracking, we create a placeholder
                        # In a production system, you might want to use a separate STT service
                        # to get the actual transcription for history
                        user_speech_placeholder = "[Audio input received]"
                        
                        # Store the conversation turn
                        self._conversation_history.append({
                            "user": user_speech_placeholder,
                            "assistant": full_response.strip()
                        })
                        
                        logger.info(f"Stored conversation turn. Total history: {len(self._conversation_history)} turns")
                        logger.info(f"Assistant response: {full_response[:100]}...")
                        
                        # Manage history size
                        self._manage_conversation_history()
                    
                    # Reset error count on successful generation
                    self._error_count = 0

                except Exception as e:
                    logger.error(f"Error generating text from model: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    
                    # Increment error count
                    self._error_count += 1
                    
                    # If too many consecutive errors, reset conversation history
                    if self._error_count >= 3:
                        logger.warning(f"Too many consecutive errors ({self._error_count}), resetting conversation history")
                        self._conversation_history = []
                        self._error_count = 0
                    
                    # Provide a fallback response
                    fallback_response = "மன்னிக்கவும், ஏதோ technical issue வந்துச்சு. மறுபடியும் சொல்லுங்க?"
                    from pipecat.frames.frames import LLMTextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
                    
                    yield LLMFullResponseStartFrame()
                    yield LLMTextFrame(text=fallback_response)
                    yield LLMFullResponseEndFrame()
                    
            else:
                logger.error("No model available for text generation")
                from pipecat.frames.frames import ErrorFrame
                yield ErrorFrame("No model available for text generation")

        except Exception as e:
            logger.error(f"Error in audio buffer processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            from pipecat.frames.frames import ErrorFrame
            yield ErrorFrame(f"Error processing audio: {str(e)}")
        finally:
            self._buffer.is_processing = False
            self._buffer.frames = []
            self._buffer.started_at = None


# ---------------- Indic Parler TTS (streaming on LLM text chunks) ----------------
class IndicParlerTTSService(FrameProcessor):
    """Minimal TTS processor that turns LLM text chunks into PCM16 audio frames.

    - Uses ai4bharat/indic-parler-tts
    - Generates per text chunk to enable streaming playback
    - Resamples model output to 16 kHz mono PCM16 for telephony
    """

    def __init__(self, model_name: str = "ai4bharat/indic-parler-tts", device: str | None = None, speaker_description: str | None = None):
        super().__init__()
        self.device = device or ("cuda:0" if (hasattr(np, "__version__") and __import__("torch").cuda.is_available()) else "cpu")

        # Load TTS model/tokenizers
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )

        # Voice prompt/description
        self.description = speaker_description or (
            "Tamil female Jaya, a friendly native Tamil speaker. Warm, expressive, "
            "clear tone, fast pitch, natural rhythm, studio-quality close mic recording."
        )

        # Cache sampling rate
        self.model_sampling_rate = int(getattr(self.model.config, "sampling_rate", 24000))

    def _tts_generate_audio(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)

        description_inputs = self.description_tokenizer(self.description, return_tensors="pt").to(self.device)
        prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        import torch
        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                max_new_tokens=70000,
            )

        audio = generation.cpu().numpy().squeeze().astype(np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)

        # Trim slight tail as in example
        trim = int(len(audio) * 0.02)
        if trim > 0:
            audio = audio[:-trim]
        return audio

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
        if src_rate == 16000 or audio.size == 0:
            return audio
        # Simple linear resample to 16 kHz to avoid extra deps
        src_len = audio.shape[0]
        dst_len = int(round(src_len * 16000 / src_rate))
        if dst_len <= 0:
            return np.zeros(0, dtype=np.float32)
        x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
        xi = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        return np.interp(xi, x, audio).astype(np.float32)

    @staticmethod
    def _float32_to_int16(audio_f32: np.ndarray) -> np.ndarray:
        if audio_f32.size == 0:
            return np.zeros(0, dtype=np.int16)
        audio_f32 = np.clip(audio_f32, -1.0, 1.0)
        return (audio_f32 * 32767.0).astype(np.int16)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Pass-through non-LLM frames
        if isinstance(frame, LLMFullResponseStartFrame):
            return []
        if isinstance(frame, LLMFullResponseEndFrame):
            return []
        if isinstance(frame, LLMTextFrame):
            # Generate per-chunk audio
            text_chunk = frame.text
            audio_f32 = self._tts_generate_audio(text_chunk)
            audio_16k_f32 = self._resample_to_16k(audio_f32, self.model_sampling_rate)
            pcm16 = self._float32_to_int16(audio_16k_f32)
            if pcm16.size == 0:
                return []
            return [AudioRawFrame(audio=pcm16, sample_rate=16000)]
        return []

# Initialize Ultravox LLM processor with system prompt and performance optimizations
ultravox_llm = UltravoxLLMService(
    model_name="fixie-ai/ultravox-v0_6-gemma-3-27b",
    hf_token=os.getenv("HF_TOKEN"),
    system_prompt=(
        "MOST IMPORTANT: Talk in Colloquial Tamil with a mixture of Tamil and English words.\n"
        "Speak in an EXTREMELY CONCISE manner.\n"
        "Use TAMIL literals for generating Tamil words and English literals for English words.\n\n"

        "You are a helpful AI assistant in a phone call. Your goal is to demonstrate "
        "your capabilities in a succinct way. Keep your responses concise and natural "
        "for voice conversation. Don't include special characters in your answers. "
        "Respond to what the user said in a creative and helpful way.\n\n"

        "IMPORTANT - CONVERSATION MEMORY:\n"
        "- You can see previous conversation turns in the context above.\n"
        "- ALWAYS remember and reference information from earlier in the conversation.\n"
        "- Use context naturally to provide relevant, connected responses.\n"
        "- If the user asks about something mentioned earlier, recall it accurately.\n"
        "- Treat the entire call as ONE continuous conversation.\n\n"

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
        "Assistant: சும்மா தான், coffee குடிக்கறேன். நீ என்ன பண்ணுறே?\n\n"

        "Example 2:\n"
        "User: Can you explain what AI means?\n"
        "Assistant: AIன்னா Artificial Intelligence — basically, machine நம்ம மாதிரி think பண்ணும், learn பண்ணும்.\n\n"

        "Example 3:\n"
        "User: Weather எப்படி இருக்கு அங்கே?\n"
        "Assistant: இங்க நாறா சூடா இருக்கு, fan full speedல போடணும் போல இருக்கு!\n\n"

        "Example 4:\n"
        "User: Tell me a joke.\n"
        "Assistant: சரி, ஓன்னு கேள் — ஒரு computerக்கு fever வந்தா, அது சொல்லும் I've got a virus! ஹா ஹா!\n\n"

        "Example 5:\n"
        "User: Can you help me with my project?\n"
        "Assistant: சொல்லு என்ன project. நம்ம சேர்ந்து பண்ணலாம் easyஆ.\n\n"

        "Remember: Mix Tamil and English naturally, keep it friendly and human, like a real phone chat between buddies.\n\n"

        "MOST VERY VERY IMPORTANT: The TAMIL should be MORE in your response THAN ENGLISH!!!!\n\n"
        "REMEMBER CAREFULLY: DO NOT EVER add a translating English phrase next to the colloquial tamil response you have generated.\n\n"
    ),
    temperature=0.6,
    max_tokens=200,  # Increased slightly for better responses
    max_conversation_turns=10,  # Keep last 10 turns
    gpu_memory_utilization=0.85,  # Slightly reduced to prevent OOM
    max_model_len=4096,
    dtype="bfloat16",
    enforce_eager=False,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
)

async def run_bot(transport: BaseTransport, handle_sigint: bool):
    # Initialize Indic Parler TTS (replaces ElevenLabs)
    tts = IndicParlerTTSService(
        model_name="ai4bharat/indic-parler-tts",
        device=None,
        speaker_description=(
            "Tamil female Jaya, a friendly native Tamil speaker. Warm, expressive, "
            "clear tone, fast pitch, natural rhythm, studio-quality close mic recording."
        ),
    )

    # Create pipeline with Ultravox as the central multimodal LLM
    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from Exotel
            ultravox_llm,  # Ultravox processes audio and generates intelligent responses
            tts,  # Text-To-Speech (Indic Parler TTS per LLMTextFrame chunk)
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

    # IMPROVED VAD settings - less aggressive to avoid cutting off speech
    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5,  # Increased from 0.2 to avoid cutting off speech
                    min_volume=0.6,  # Lower threshold for better sensitivity
                )
            ),
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
