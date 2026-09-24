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
    """
    Converts AI's text response back to speech using completely free gTTS (Google TTS).
    The api_key argument is kept for compatibility but ignored here.
    """
    try:
        # 'bn' stands for Bengali
        tts = gTTS(text=text, lang='bn', slow=False)
        tts.save(save_path)
    except Exception as e:
        print("TTS Error:", e)
        
    return save_path

