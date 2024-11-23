from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from pathlib import Path
import soundfile as sf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from f5_tts.api import F5TTS

app = FastAPI(
    title="F5-TTS API",
    description="Text-to-Speech API using F5-TTS model",
    version="1.0.0"
)

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
        vocoder_name="vocos",
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>F5-TTS API</title>
        </head>
        <body>
            <h1>F5-TTS API</h1>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><a href="/health">/health</a> - Health check endpoint</li>
                <li><code>/tts/generate</code> - Text-to-Speech generation endpoint (POST)</li>
            </ul>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """
    Check if the API is running and models are loaded
    """
    return {
        "status": "healthy",
        "models_loaded": tts is not None,
        "device": tts.device if tts else None
    }

@app.post("/tts/generate")
async def generate_tts(
    reference_audio: UploadFile = File(..., description="Reference audio file (WAV format)"),
    reference_text: str = Form(..., description="Text content of the reference audio"),
    text: str = Form(..., description="Text to convert to speech"),
    remove_silence: bool = Form(False, description="Whether to remove silence from the generated audio"),
    speed: float = Form(1.0, description="Speech speed multiplier (1.0 = normal speed)"),
):
    """
    Generate speech from text using F5-TTS model
    
    - Upload a reference audio file and its corresponding text
    - Provide the text you want to convert to speech
    - Optionally adjust speed and silence removal
    
    Returns a WAV audio file
    """
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