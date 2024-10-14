import asyncio
import io
import os
import uuid
import wave

import aiohttp
import pyaudio

# import requests
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL")
sequence_id = 0


class AudioStreamer:
    def __init__(self, chunk_size=1024, sample_rate=16000, record_seconds=5):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()

    def stream_audio(self):
        # print("Starting audio stream...")

        stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        # print("Streaming audio...")
        while True:
            frames = []
            for _ in range(
                0, int(self.sample_rate / self.chunk_size * self.record_seconds)
            ):
                data = stream.read(self.chunk_size)
                frames.append(data)

            audio_data = b"".join(frames)
            yield audio_data

    def close(self):
        self.p.terminate()


def save_wav(audio_data, sample_rate=16000, channels=1):
    # print("Saving WAV...")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    return buffer.getvalue()


async def send_audio_for_transcription(audio_data, endpoint_url):
    # print(f"Sending request to {endpoint_url}")

    global sequence_id
    wav_data = save_wav(audio_data)
    filename = f"audio_{sequence_id:04d}.wav"
    sequence_id += 1

    # # Save the WAV file
    # filename = f"audio_{uuid.uuid4()}.wav"
    # with open(filename, "wb") as f:
    #     f.write(wav_data)
    #     f.close()

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("file", wav_data, filename=filename, content_type="audio/wav")

        try:
            async with session.post(endpoint_url, data=data) as response:
                response.raise_for_status()
                result = await response.json()
                result["sequence_id"] = sequence_id
            return result
        except aiohttp.ClientError as e:
            print(f"Error sending request: {e}")
            return None


async def process_audio_stream(streamer, endpoint_url):
    for audio_chunk in streamer.stream_audio():
        task = asyncio.create_task(
            send_audio_for_transcription(audio_chunk, endpoint_url)
        )
        try:
            result = await asyncio.wait_for(task, timeout=5.0)  # 5 second timeout
            if result:
                print(
                    f"Sequence: {result['sequence_id']}, Transcription: {result.get('transcription', 'No transcription available')}"
                )
            else:
                print("Failed to get transcription")
        except asyncio.TimeoutError:
            print("Transcription request timed out")
        except Exception as e:
            print(f"Error processing transcription: {e}")


async def main():
    endpoint_url = os.getenv("TRANSCRIPTION_URL", "http://localhost:8000/transcribe")
    streamer = AudioStreamer()

    try:
        await process_audio_stream(streamer, endpoint_url)
    except KeyboardInterrupt:
        print("Stopping audio stream...")
    finally:
        streamer.close()


if __name__ == "__main__":
    asyncio.run(main())
