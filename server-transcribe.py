import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from nemo.collections.asr.models import EncDecMultiTaskModel


app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        print(f"File saved temporarily as: {temp_file_path}")
    
        transcription = canary_model.transcribe(
            paths2audio_files=[temp_file_path],
            batch_size=16,
        )        
                
        print(f"Transcription completed: {transcription}")
        
        return JSONResponse(content={"transcription": transcription})
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file removed: {temp_file_path}")

if __name__ == "__main__":

    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
    
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    uvicorn.run(app, host="0.0.0.0", port=8726)
