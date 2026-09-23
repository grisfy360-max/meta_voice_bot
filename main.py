import os
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Meta Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "my_secure_verify_token_123")
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Meta Voice Assistant Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; color: #333; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .status-box { background: #e8f8f5; border-left: 5px solid #1abc9c; padding: 15px; margin: 20px 0; border-radius: 4px; }
            .btn { background: #3498db; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer; transition: 0.3s; }
            .btn:hover { background: #2980b9; }
            .log-box { background: #2c3e50; color: #ecf0f1; padding: 15px; height: 200px; overflow-y: scroll; border-radius: 5px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Meta Voice Assistant Dashboard</h1>
            <div class="status-box">
                <strong>Status:</strong> 🟢 Server is running perfectly!<br>
                <strong>Webhook URL:</strong> /webhook
            </div>
            
            <h3>Live Logs</h3>
            <div class="log-box" id="logs">
                > System initialized...<br>
                > Waiting for Meta Webhook events...<br>
            </div>
            
            <br>
            <button class="btn" onclick="alert('Manual testing will be enabled in Phase 4!')">▶ Test Bot Manually</button>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 1. Webhook Verification (Meta will call this during setup)
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("Webhook Verified successfully!")
            return Response(content=challenge, status_code=200)
    return Response(content="Verification failed", status_code=403)

from audio_services import download_audio_from_meta, speech_to_text, text_to_speech
from langchain_agent import process_text_with_ai
import json

# Background task to process incoming voice messages
def process_voice_message(data):
    print("Processing incoming data in background...")
    
    # Meta Webhook payload parsing (Assuming WhatsApp Cloud API format)
    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        # Check if it's a message
        if "messages" in value:
            message = value["messages"][0]
            phone_number_id = value["metadata"]["phone_number_id"]
            sender_id = message["from"]
            
            # If the user sent an audio message (Voice Note)
            if message.get("type") == "audio":
                media_id = message["audio"]["id"]
                print(f"Received audio from {sender_id}. Media ID: {media_id}")
                
                # 1. Download Audio
                input_audio_path = f"incoming_{media_id}.ogg"
                download_audio_from_meta(media_id, ACCESS_TOKEN, input_audio_path)
                
                # 2. Speech to Text (STT) using Gemini
                print("Converting Speech to Text using Gemini...")
                user_text = speech_to_text(input_audio_path, os.getenv("GEMINI_API_KEY"))
                print(f"User said: {user_text}")
                
                # 3. AI Brain (LangChain)
                print("Generating AI Response...")
                ai_reply = process_text_with_ai(user_text, user_id=sender_id)
                print(f"AI Output: {ai_reply}")
                
                # 4. Text to Speech (TTS)
                print("Converting Text to Speech...")
                output_audio_path = f"reply_{media_id}.mp3"
                text_to_speech(ai_reply, os.getenv("ELEVENLABS_API_KEY"), output_audio_path)
                
                # 5. Send Audio Reply back to User via Meta API
                send_audio_reply_whatsapp(phone_number_id, sender_id, output_audio_path)
                
                # Cleanup local files
                if os.path.exists(input_audio_path): os.remove(input_audio_path)
                if os.path.exists(output_audio_path): os.remove(output_audio_path)
                
            else:
                print("Received a non-audio message. Ignoring.")
    except Exception as e:
        print(f"Error processing message: {e}")

def send_audio_reply_whatsapp(phone_number_id, recipient_id, audio_path):
    """Uploads the generated audio and sends it via WhatsApp Business API."""
    import requests
    
    # URL for uploading media
    upload_url = f"https://graph.facebook.com/v18.0/{phone_number_id}/media"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    print("Uploading audio to Meta...")
    with open(audio_path, "rb") as f:
        files = {
            "file": (audio_path, f, "audio/mpeg"),
            "type": (None, "audio"),
            "messaging_product": (None, "whatsapp")
        }
        upload_res = requests.post(upload_url, headers=headers, files=files)
        
    if upload_res.status_code == 200:
        media_id = upload_res.json().get("id")
        
        # Send the uploaded media to the user
        send_url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "audio",
            "audio": {"id": media_id}
        }
        send_res = requests.post(send_url, headers=headers, json=payload)
        if send_res.status_code == 200:
            print("Successfully sent voice reply to user!")
        else:
            print("Failed to send message:", send_res.text)
    else:
        print("Failed to upload media:", upload_res.text)

# 2. Receive Messages from Meta
@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        
        # We send the processing to a background task so Meta gets a fast 200 OK
        background_tasks.add_task(process_voice_message, data)
        
    except Exception as e:
        print(f"Error parsing webhook data: {e}")

    return Response(content="EVENT_RECEIVED", status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
