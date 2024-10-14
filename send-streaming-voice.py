import io
import os
import uuid
import wave

import pyaudio
import requests
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL")


class AudioStreamer:
    def __init__(self, chunk_size=1024, sample_rate=16000, record_seconds=5):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()

    def stream_audio(self):
        stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        print("Streaming audio...")
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
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    return buffer.getvalue()


def send_audio_for_transcription(audio_data, endpoint_url):
    wav_data = save_wav(audio_data)
    files = {"file": ("audio.wav", wav_data, "audio/wav")}

    # Save the WAV file
    filename = f"audio_{uuid.uuid4()}.wav"
    with open(filename, "wb") as f:
        f.write(wav_data)
        f.close()

    try:
        response = requests.post(endpoint_url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error sending request: {e}")
        return None


def main():
    streamer = AudioStreamer()

    try:
        for audio_chunk in streamer.stream_audio():
            result = send_audio_for_transcription(audio_chunk, TRANSCRIPTION_URL)
            if result:
                print(
                    "Transcription:",
                    result.get("transcription", "No transcription available"),
                )
            else:
                print("Failed to get transcription")
    except KeyboardInterrupt:
        print("Stopping audio stream...")
    finally:
        streamer.close()


if __name__ == "__main__":
    main()
