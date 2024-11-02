import asyncio
import io
import logging
import os
import wave
from collections import deque

import aiohttp
import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL", "http://localhost:8000/transcribe")
sequence_id = 0


class AudioStreamer:
    def __init__(
        self,
        chunk_size=512,  # Smaller chunk size for more responsive VAD
        sample_rate=16000,
        max_buffer_duration=10,
        min_speech_duration_ms=600,
        min_silence_duration_ms=400,
        window_size_samples=512,
        speech_pad_ms=30,
        vad_threshold=0.5,
    ):
        """
        Initialize AudioStreamer with Silero VAD configuration

        Args:
            chunk_size: Size of audio chunks to process
            sample_rate: Audio sample rate (must be 16000 for Silero VAD)
            max_buffer_duration: Maximum duration of audio buffer in seconds
            min_speech_duration_ms: Minimum speech duration in milliseconds
            min_silence_duration_ms: Minimum silence duration in milliseconds
            window_size_samples: Size of window for VAD processing
            speech_pad_ms: Padding around speech segments in milliseconds
            vad_threshold: VAD threshold for speech detection
        """
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.max_buffer_duration = max_buffer_duration
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()

        # VAD parameters
        self.min_speech_samples = int(min_speech_duration_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)
        self.window_size_samples = window_size_samples
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        self.vad_threshold = vad_threshold

        # Initialize Silero VAD
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.vad_model.eval()

        # Buffer for collecting audio frames
        self.audio_buffer = deque(maxlen=int(sample_rate * max_buffer_duration))
        self.speech_frames = []

    def _get_speech_timestamps(self, audio_tensor):
        """Get speech timestamps using Silero VAD"""
        return self.vad_model(
            audio_tensor,
            self.sample_rate,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            window_size_samples=self.window_size_samples,
            speech_pad_ms=30,
            return_seconds=False,
        )

    def _is_speech(self, audio_chunk):
        """Check if audio chunk contains speech using Silero VAD"""
        # Convert audio to float32 and normalize
        audio_tensor = torch.FloatTensor(audio_chunk).unsqueeze(0)
        audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))

        # Get speech probability
        speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        return speech_prob > self.vad_threshold

    async def stream_audio(self):
        """Stream audio with VAD-based segmentation"""
        stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        try:
            speech_detected = False
            silence_counter = 0

            while True:
                # Read audio chunk
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # Add chunk to buffer
                self.audio_buffer.extend(audio_chunk)

                # Check for speech in current chunk
                is_speech = self._is_speech(audio_chunk)

                if is_speech and not speech_detected:
                    # Speech started
                    speech_detected = True
                    silence_counter = 0
                    self.speech_frames = [data]
                elif speech_detected:
                    self.speech_frames.append(data)
                    if not is_speech:
                        silence_counter += len(audio_chunk)
                        if silence_counter >= self.min_silence_samples:
                            # Speech ended, yield the collected frames
                            if (
                                len(self.speech_frames) * self.chunk_size
                                >= self.min_speech_samples
                            ):
                                audio_data = b"".join(self.speech_frames)
                                yield audio_data
                            # Reset for next utterance
                            speech_detected = False
                            self.speech_frames = []
                            silence_counter = 0

                await asyncio.sleep(0)  # Yield control

        finally:
            stream.stop_stream()
            stream.close()

    def close(self):
        self.p.terminate()


def save_wav(audio_data, sample_rate=16000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    return buffer.getvalue()


async def send_audio_for_transcription(audio_data, endpoint_url, seq_id):
    wav_data = save_wav(audio_data)
    filename = f"audio_{seq_id:04d}.wav"

    data = aiohttp.FormData()
    data.add_field("file", wav_data, filename=filename, content_type="audio/wav")

    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            logging.debug(f"Sending chunk {seq_id}...")

            async with session.post(endpoint_url, data=data) as response:
                logging.debug(f"Received status {response.status} for chunk {seq_id}")
                response.raise_for_status()
                result = await response.json()

                logging.debug(f"Result: {seq_id}: {result}")
                return result
    except aiohttp.ClientError as e:
        print(f"Error sending chunk {seq_id}: {e}")
        return None


async def audio_producer(queue, streamer):
    try:
        async for audio_chunk in streamer.stream_audio():
            logging.debug("Adding audio chunk to queue...")
            await queue.put(audio_chunk)
            await asyncio.sleep(0)  # Yield control to allow other tasks to run
    except Exception as e:
        print(f"Producer error: {e}")
    finally:
        await queue.put(None)  # Send sentinel to stop consumer


async def audio_consumer(queue, endpoint_url):
    global sequence_id

    while True:
        logging.debug("Waiting to get audio chunk from queue...")
        audio_chunk = await queue.get()
        if audio_chunk is None:
            logging.debug("Received sentinel. Exiting consumer.")
            break

        seq_id = sequence_id
        sequence_id += 1

        logging.debug(f"Processing chunk {seq_id}...")
        try:
            result = await send_audio_for_transcription(
                audio_chunk, endpoint_url, seq_id
            )
            if result:
                print(f"{seq_id}: {result.get('transcription', 'No transcription')}")
            else:
                print(f"Failed to get transcription for chunk {seq_id}.")
        except Exception as e:
            print(f"Error processing chunk {seq_id}: {e}")
        finally:
            queue.task_done()


async def main():
    streamer = AudioStreamer()
    queue = asyncio.Queue()

    logging.debug("Starting producer and consumer tasks...")
    producer_task = asyncio.create_task(audio_producer(queue, streamer))
    consumer_task = asyncio.create_task(audio_consumer(queue, TRANSCRIPTION_URL))
    logging.info("Started transcription tasks...")

    await asyncio.gather(producer_task, consumer_task)

    streamer.close()
    print("Streaming and transcription completed.")


if __name__ == "__main__":
    asyncio.run(main())
