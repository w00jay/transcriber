# Real-time Audio Transcription System

This project implements a near real-time audio transcription system with a GPU-backed transcription server and a client for continuous audio streaming and transcription.


## Components

1. **Server (server-transcribe.py)**: Runs on a CUDA GPU-enabled machine, providing transcription services via HTTP.
2. **Client (send-streaming-voice.py)**: Captures audio and sends it to the server for real-time transcription.


## Server: server-transcribe.py

### Requirements
- Python 3.10+
- FastAPI
- uvicorn
- NVIDIA GPU with CUDA support
- NeMo ASR model (Canary-1B)

### Setup
1. Install required packages:
   ```
   pip install fastapi uvicorn nemo_toolkit[all] python-dotenv
   ```
2. Ensure necessary CUDA libraries are installed for your GPU.

### Running the Server

```
python server-transcribe.py
```


## Client: send-streaming-voice(-silero).py

### Requirements
- Python 3.7+
- pyaudio
- requests
- python-dotenv

### Setup
1. Install required packages:
   ```
   pip install pyaudio requests python-dotenv
   ```

### Configuration
Create a `.env` file in the same directory as `send-streaming-voice.py`:

```
TRANSCRIBE_ENDPOINT=http://your-server-ip:8726/transcribe
```

Replace `your-server-ip` with the IP address or hostname of your GPU server.

### Running the Client

```
python send-streaming-voice.py
```

### Evaluation

```
python text-diff.py test/data/text-source.txt test/data/text-transcribed(-silero).txt
```

## Usage

1. Start the server on your GPU-enabled machine.
2. Run the client on the machine where you want to capture audio.
3. Speak into the microphone connected to the client machine.
4. The client will continuously send audio chunks to the server.
5. The server will process these chunks and return transcriptions.
6. The client will print the transcriptions as they are received.


## System Architecture

```
[Client Machine]                 [GPU Server]
+------------------+             +------------------+
|                  |             |                  |
| Microphone       |             | NVIDIA GPU       |
|      |           |             |      |           |
|      v           |             |      v           |
| send-streaming-  |  HTTP POST  | server-          |
| voice.py         | --------->  | transcribe.py    |
|      |           | Audio Data  |      |           |
|      |           |             |      |           |
|      |           | HTTP        |      |           |
|      |           | Response    |      |           |
|      v           | <---------  |      |           |
| Display          |Transcription|      |           |
| Transcription    |             |      |           |
+------------------+             +------------------+
```

## Implementation Details

### Server (server-transcribe.py)
- Uses FastAPI to create an HTTP server.
- Loads a pre-trained NeMo ASR model (Canary-1B) for transcription.
- Receives audio chunks via POST requests.
- Processes audio using the GPU for fast transcription.
- Returns transcribed text as JSON responses.

### Client (send-streaming-voice.py)
- Uses PyAudio to capture audio from the microphone.
- Streams audio in chunks (default 5 seconds).
- Sends each audio chunk to the server as a WAV file in a POST request.
- Receives and displays the transcription for each chunk.

## Notes

- The server's firewall should allow incoming connections on the specified port.
- Audio is captured in 5-second chunks by default. Adjust this in the `AudioStreamer` class if needed.
- The transcription model used is NVIDIA's Canary-1B. You may need to adjust paths or model loading based on your specific setup.

## Troubleshooting

- For audio capture issues, ensure your microphone is properly configured and recognized by your system.
- For server connection issues, verify that the `TRANSCRIBE_ENDPOINT` in the client's `.env` file is correct and that the server is running and accessible.
- GPU-related issues on the server side may require checking CUDA installation and compatibility with the NeMo toolkit.


## License
MIT License
