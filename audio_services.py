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



def text_to_speech(text: str, api_key: str, save_path: str):
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
            
        persona = cfg.get("persona", "You are a helpful assistant.")
        voice_name = cfg.get("voice", "Puck")
        
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
