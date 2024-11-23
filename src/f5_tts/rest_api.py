from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from pathlib import Path
import soundfile as sf
import sys
import requests
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from f5_tts.api import F5TTS

app = FastAPI(
    title="F5-TTS API",
    description="Text-to-Speech API using F5-TTS model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {
        "status": "healthy",
        "models_loaded": tts is not None,
        "device": tts.device if tts else None
    }

def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.unlink(path)
    except Exception as e:
        print(f"Error cleaning up file {path}: {e}")

async def download_audio(url: str) -> str:
    """Download audio from URL and save to temporary file"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        ref_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        ref_path = ref_file.name
        ref_file.write(response.content)
        ref_file.close()
        return ref_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")

@app.post("/tts/generate")
async def generate_tts(
    reference_audio: Optional[UploadFile] = File(None, description="Reference audio file (WAV format)"),
    reference_audio_url: Optional[str] = Form(None, description="URL to reference audio file (WAV format)"),
    reference_text: str = Form("", description="Text content of the reference audio (leave empty for auto-transcription)"),
    text: str = Form(..., description="Text to convert to speech"),
    remove_silence: bool = Form(False, description="Whether to remove silence from the generated audio"),
    speed: float = Form(1.0, description="Speech speed multiplier (1.0 = normal speed)"),
):
    if not tts:
        raise HTTPException(status_code=500, detail="TTS model not initialized")
    
    if not reference_audio and not reference_audio_url:
        raise HTTPException(status_code=400, detail="Either reference_audio or reference_audio_url must be provided")
    
    ref_path = None
    output_path = None
    
    try:
        # Handle reference audio from either file upload or URL
        if reference_audio:
            ref_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            ref_path = ref_file.name
            ref_file.write(await reference_audio.read())
            ref_file.close()
        else:
            ref_path = await download_audio(reference_audio_url)

        # Create temporary output file
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        output_path = out_file.name
        out_file.close()

        # Generate audio - F5TTS will handle transcription automatically if reference_text is empty
        wav, sr, _ = tts.infer(
            ref_file=ref_path,
            ref_text=reference_text,
            gen_text=text,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=output_path
        )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_audio.wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        if ref_path:
            cleanup_file(ref_path)
        if output_path:
            try:
                import asyncio
                asyncio.create_task(
                    asyncio.sleep(1).then(lambda: cleanup_file(output_path))
                )
            except Exception:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)