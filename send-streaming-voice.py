import asyncio
import io
import logging
import os
import wave

import aiohttp
import pyaudio
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL", "http://localhost:8000/transcribe")
sequence_id = 0


class AudioStreamer:
    def __init__(self, chunk_size=1024, sample_rate=16000, buffer_duration=10):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration  # seconds per chunk
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()

    async def stream_audio(self):
        stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        try:
            while True:
                frames = []
                for _ in range(
                    int(self.sample_rate / self.chunk_size * self.buffer_duration)
                ):
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    await asyncio.sleep(0)  # Allow other tasks to run

                audio_data = b"".join(frames)
                yield audio_data  # Generate an audio chunk
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
        timeout = aiohttp.ClientTimeout(total=30)
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
