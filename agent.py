from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import whisper
import pyttsx3
import io
import tempfile
import asyncio
import logging
from typing import Optional, List, Dict, Any
import json
import uuid
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import wave

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="English Tutor Voice AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded once at startup)
whisper_model = None
tts_engine = None

# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class SessionConfig(BaseModel):
    proficiency_level: str = "intermediate"
    scenario: Optional[str] = None
    system_prompt: Optional[str] = None

class TextRequest(BaseModel):
    session_id: str
    message: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global whisper_model, tts_engine
    
    logger.info("🚀 Starting English Tutor Backend...")
    
    # Initialize Whisper
    logger.info("📥 Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    
    # Initialize TTS
    logger.info("🔊 Initializing TTS engine...")
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.9)
    
    # Check Ollama
    try:
        models_response = ollama.list()
        logger.info("✅ Ollama is running")
        
        # Debug: Print the actual response structure
        logger.info(f"📊 Ollama response structure: {models_response}")
        
        # Check if phi3:3.8b exists - with better error handling
        model_names = []
        if isinstance(models_response, dict) and 'models' in models_response:
            for model in models_response['models']:
                if isinstance(model, dict):
                    # Try different possible keys for model name
                    name = model.get('name') or model.get('model') or model.get('id', 'unknown')
                    model_names.append(name)
                    logger.info(f"📦 Found model: {name}")
                else:
                    logger.warning(f"⚠️ Unexpected model format: {model}")
        else:
            logger.warning(f"⚠️ Unexpected response format from ollama.list(): {models_response}")
        
        target_model = 'phi3:3.8b'
        if target_model not in model_names:
            logger.info(f"📥 Model {target_model} not found. Downloading...")
            try:
                ollama.pull(target_model)
                logger.info(f"✅ Successfully downloaded {target_model}")
            except Exception as pull_error:
                logger.error(f"❌ Failed to download {target_model}: {pull_error}")
        else:
            logger.info(f"✅ Model {target_model} is available")
            
    except Exception as e:
        logger.error(f"❌ Ollama error during startup: {e}")
        logger.info("💡 Make sure Ollama is running: 'ollama serve' in terminal")

def build_system_prompt(proficiency_level: str = "intermediate", scenario: Optional[str] = None) -> str:
    """Build system prompt based on configuration"""
    base_prompt = f"""You are a friendly and encouraging English tutor.

Guidelines:
- The learner's proficiency level is {proficiency_level}
- Adjust vocabulary and complexity for {proficiency_level} level
- Correct mistakes gently and provide better alternatives
- Give encouragement and positive feedback  
- Ask follow-up questions to keep conversations flowing
- Keep responses concise (2-3 sentences max) for better conversation flow
- Be patient and supportive"""
    
    if scenario:
        base_prompt += f"\n\nScenario: Role-play as someone in '{scenario}'. Act naturally while being helpful as a tutor."
    
    return base_prompt

@app.post("/session/create")
async def create_session(config: SessionConfig) -> Dict[str, str]:
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    
    system_prompt = config.system_prompt or build_system_prompt(
        config.proficiency_level, 
        config.scenario
    )
    
    sessions[session_id] = {
        "conversation_history": [
            {"role": "system", "content": system_prompt}
        ],
        "config": config.dict()
    }
    
    logger.info(f"📝 Created session {session_id[:8]}... with {config.proficiency_level} level")
    
    return {
        "session_id": session_id,
        "status": "created",
        "message": f"Session created for {config.proficiency_level} level English practice"
    }

@app.post("/chat/text")
async def chat_text(request: TextRequest) -> SessionResponse:
    """Process text message and return text response"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    
    # Add user message to history
    session["conversation_history"].append({
        "role": "user", 
        "content": request.message
    })
    
    try:
        # Generate response with Ollama
        response = ollama.chat(
            model="phi3:3.8b",
            messages=session["conversation_history"],
            options={
                "temperature": 0.8,
                "top_p": 0.9,
            }
        )
        
        ai_response = response['message']['content']
        
        # Add AI response to history
        session["conversation_history"].append({
            "role": "assistant",
            "content": ai_response
        })
        
        # Keep conversation history manageable (last 20 messages + system)
        if len(session["conversation_history"]) > 21:
            system_msg = session["conversation_history"][0]
            session["conversation_history"] = [system_msg] + session["conversation_history"][-20:]
        
        logger.info(f"💬 Text chat for session {request.session_id[:8]}...")
        
        return SessionResponse(
            session_id=request.session_id,
            message=ai_response
        )
        
    except Exception as e:
        logger.error(f"❌ Ollama error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

@app.post("/chat/voice")
async def chat_voice(session_id: str, audio: UploadFile = File(...)):
    """Process voice message and return text response"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be audio")
    
    session = sessions[session_id]
    
    try:
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Transcribe with Whisper
        result = whisper_model.transcribe(temp_path)
        user_text = result["text"].strip()
        
        # Clean up temp file
        Path(temp_path).unlink()
        
        if not user_text or len(user_text) < 3:
            raise HTTPException(status_code=400, detail="Could not understand audio")
        
        # Process as text message
        text_request = TextRequest(session_id=session_id, message=user_text)
        response = await chat_text(text_request)
        
        # Add transcribed text to response
        return {
            **response.dict(),
            "transcribed_text": user_text
        }
        
    except Exception as e:
        logger.error(f"❌ Voice processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process voice message")

@app.post("/tts/generate")
async def generate_speech(session_id: str, text: str):
    """Generate speech from text"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Create temporary file for TTS output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Generate speech with pyttsx3
        tts_engine.save_to_file(text, temp_path)
        tts_engine.runAndWait()
        
        # Return audio file
        def iterfile():
            with open(temp_path, mode="rb") as file_like:
                yield from file_like
            # Clean up after streaming
            Path(temp_path).unlink()
        
        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate speech")

@app.post("/chat/voice-to-voice")
async def voice_to_voice(session_id: str, audio: UploadFile = File(...)):
    """Complete voice pipeline: voice input → text response → speech output"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Step 1: Voice to text
        voice_response = await chat_voice(session_id, audio)
        response_text = voice_response["message"]
        
        # Step 2: Text to speech
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        tts_engine.save_to_file(response_text, temp_path)
        tts_engine.runAndWait()
        
        # Return audio response
        def iterfile():
            with open(temp_path, mode="rb") as file_like:
                yield from file_like
            Path(temp_path).unlink()
        
        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav",
                "X-Transcribed-Text": voice_response["transcribed_text"],
                "X-Response-Text": response_text
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Voice-to-voice error: {e}")
        raise HTTPException(status_code=500, detail="Voice processing failed")

@app.get("/session/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    # Return history without system message
    history = [msg for msg in session["conversation_history"] if msg["role"] != "system"]
    
    return {
        "session_id": session_id,
        "history": history,
        "config": session["config"]
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama with better error handling
        models_response = ollama.list()
        ollama_status = "running"
        
        # Count available models
        model_count = 0
        if isinstance(models_response, dict) and 'models' in models_response:
            model_count = len(models_response['models'])
            
    except Exception as e:
        ollama_status = f"error: {str(e)}"
        model_count = 0
    
    return {
        "status": "healthy",
        "whisper": "loaded" if whisper_model else "error",
        "tts": "loaded" if tts_engine else "error", 
        "ollama": ollama_status,
        "ollama_models": model_count,
        "active_sessions": len(sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)