import os
import requests

def download_audio_from_meta(media_id: str, access_token: str, save_path: str):
    """
    Downloads an audio file (e.g., OGG voice note) from Meta using its Media ID.
    """
    # Step 1: Get the media URL from Meta Graph API
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    media_url = response.json().get("url")
    if not media_url:
        raise Exception("Media URL not found in Meta response")
        
    # Step 2: Download the actual audio content
    audio_response = requests.get(media_url, headers=headers)
    audio_response.raise_for_status()
    
    # Save the file locally
    with open(save_path, "wb") as f:
        f.write(audio_response.content)
        
    return save_path


import google.generativeai as genai
from gtts import gTTS

def speech_to_text(audio_path: str, api_key: str):
    """
    Converts downloaded audio file to text using Gemini 1.5 Flash (Free & Powerful).
    """
    genai.configure(api_key=api_key)
    try:
        # Upload the audio file to Gemini
        audio_file = genai.upload_file(path=audio_path)
        import time
        while audio_file.state.name == 'PROCESSING':
            time.sleep(1)
            audio_file = genai.get_file(audio_file.name)
        
        # Ask Gemini to transcribe it
        model = genai.GenerativeModel("gemini-3.5-flash")
        prompt = "Listen to this audio and accurately transcribe it in Bengali. Only return the text, nothing else."
        response = model.generate_content([prompt, audio_file])
        
        return response.text
    except Exception as e:
        print("STT Error:", e)
        return ""



def text_to_speech(text: str, api_key: str, save_path: str, override_voice: str = None, override_persona: str = None):
    from google import genai
    from google.genai import types
    import wave
    import json
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Load config from UI
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except:
            cfg = {}
            
        persona = override_persona if override_persona else cfg.get("persona", "You are a helpful assistant.")
        voice_name = override_voice if override_voice else cfg.get("voice", "Puck")
        
        full_prompt = f"{persona}\n\nText to speak:\n{text}"
        
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            )
        )
        
        print(f"Generating TTS with Gemini ({voice_name})...")
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-tts-preview',
                contents=full_prompt,
                config=config
            )
        except Exception as e:
            print("Quota exceeded or error with 3.1, falling back to 2.5:", e)
            response = client.models.generate_content(
                model='gemini-2.5-flash-preview-tts',
                contents=full_prompt,
                config=config
            )
            
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                with wave.open(save_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(24000)
                    wav_file.writeframes(part.inline_data.data)
                print("Gemini TTS audio saved to", save_path)
                return save_path
                
    except Exception as e:
        print("Gemini TTS Error:", e)
        # Fallback to gTTS if Gemini TTS fails
        from gtts import gTTS
        tts = gTTS(text=text, lang='bn', slow=False)
        tts.save(save_path)
        
    return save_path


import io
import json
import asyncio
from typing import AsyncGenerator
from google import genai
from google.genai import types
import edge_tts

def get_gemini_api_key() -> str:
    """Helper to retrieve the configured Gemini API key from config.json or environment."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
            key = cfg.get("gemini_api_key")
            if key:
                return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")

async def transcribe_audio_bytes(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """
    Transcribes raw audio bytes (WebM, WAV, MP3) directly in memory using Gemini Flash.
    Eliminates disk I/O, temporary files, and File API polling.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return ""
    
    api_key = get_gemini_api_key()
    if not api_key:
        print("[STT Error] No Gemini API key found.")
        return ""
        
    client = genai.Client(api_key=api_key)
    
    stt_prompt = (
        "Accurately transcribe this audio recording in Bengali (বাংলা). "
        "Return ONLY the exact transcribed text, without any introductory phrases, formatting, quotes, or notes."
    )
    
    # Normalize mime_type (strip codecs parameter for clean Gemini ingestion)
    base_mime = mime_type.split(";")[0].strip().lower()
    if base_mime not in ["audio/webm", "audio/mp4", "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/aac"]:
        base_mime = "audio/webm"

    candidate_models = [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash"
    ]
    for model_name in candidate_models:
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=base_mime),
                        stt_prompt
                    ]
                ),
                timeout=3.5
            )
            transcript = response.text.strip() if response and response.text else ""
            if transcript:
                return transcript
        except Exception as e:
            print(f"[STT Warning] Model {model_name} failed: {e}")
            continue
            
    return ""

async def stream_edge_tts(
    text: str, 
    voice: str = "bn-BD-NabanitaNeural"
) -> AsyncGenerator[bytes, None]:
    """
    Asynchronously streams synthesized Bengali speech as MP3 byte chunks.
    Falls back to in-memory gTTS if Edge-TTS service is unavailable or throttled.
    """
    if not text or not text.strip():
        return
        
    clean_text = text.replace("*", "").replace("#", "").strip()
    
    try:
        communicate = edge_tts.Communicate(clean_text, voice)
        has_audio = False
        async for chunk in communicate.stream():
            if chunk["type"] == "audio" and chunk["data"]:
                has_audio = True
                yield chunk["data"]
        if has_audio:
            return
    except Exception as edge_err:
        print(f"[TTS Alert] Edge-TTS error ({edge_err}). Activating in-memory gTTS fallback...")
        
    try:
        def generate_gtts_bytes(t: str) -> bytes:
            buf = io.BytesIO()
            tts = gTTS(text=t, lang="bn", slow=False)
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
            
        audio_bytes = await asyncio.to_thread(generate_gtts_bytes, clean_text)
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]
    except Exception as fallback_err:
        print(f"[TTS Critical] Both Edge-TTS and fallback failed: {fallback_err}")



