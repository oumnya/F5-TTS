from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from pathlib import Path
import soundfile as sf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from f5_tts.api import F5TTS

app = FastAPI(title="F5-TTS REST API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TTS instance
tts = None

@app.on_event("startup")
async def startup_event():
    global tts
    tts = F5TTS(
        model_type="F5-TTS",
        vocoder_name="vocos",  # or "bigvgan"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": tts is not None}

@app.post("/tts/generate")
async def generate_tts(
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(...),
    text: str = Form(...),
    remove_silence: bool = Form(False),
    speed: float = Form(1.0),
):
    if not tts:
        raise HTTPException(status_code=500, detail="TTS model not initialized")
    
    try:
        # Save uploaded reference audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_ref:
            temp_ref.write(await reference_audio.read())
            ref_path = temp_ref.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_out:
            output_path = temp_out.name

        # Generate audio
        wav, sr, _ = tts.infer(
            ref_file=ref_path,
            ref_text=reference_text,
            gen_text=text,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=output_path
        )

        # Clean up reference file
        os.unlink(ref_path)

        # Return the generated audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_audio.wav",
            background=os.unlink
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)