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

import json
from fastapi.responses import FileResponse

def get_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse("admin.html")

@app.get("/api/config")
def read_config():
    return get_config()

@app.get("/api/preview_voice")
def preview_voice(voice: str = "Puck"):
    import os
    import time
    from audio_services import text_to_speech
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return Response(content="Missing API Key", status_code=500)
        
    text = "হ্যালো! আমি আপনার এআই অ্যাসিস্ট্যান্ট। আমার ভয়েস ঠিক এরকম শোনাবে।"
    # Unique file name to avoid concurrency issues
    file_path = f"preview_{voice}_{int(time.time())}.wav"
    
    try:
        audio_path = text_to_speech(text, api_key, file_path, override_voice=voice)
        return FileResponse(audio_path, media_type="audio/wav")
    except Exception as e:
        return Response(content=str(e), status_code=500)

@app.post("/api/config")
async def update_config(request: Request):
    data = await request.json()
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return {"status": "success"}

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
    try:
        obj = data.get("object")
        entry = data.get("entry", [])[0]
        
        # ==========================================
        # 1. MESSENGER WEBHOOK
        # ==========================================
        if obj == "page":
            messaging = entry.get("messaging", [])[0]
            sender_id = messaging["sender"]["id"]
            message = messaging.get("message", {})
            
            # Debug the payload
            print(f"DEBUG Messenger payload: {message}")
            
            # Ignore messages sent by the bot itself (echos)
            if message.get("is_echo") or not cfg.get("messenger_active", True):
                print("Ignored echo message from bot.")
                return
            
            # Check if it has an audio attachment or text
            attachments = message.get("attachments", [])
            text_body = message.get("text")
            
            if attachments and attachments[0]["type"] == "audio":
                audio_url = attachments[0]["payload"]["url"]
                print(f"Received audio from Messenger user {sender_id}.")
                
                # Download Audio
                input_audio_path = f"incoming_{sender_id}.mp4"
                import requests
                res = requests.get(audio_url)
                with open(input_audio_path, "wb") as f:
                    f.write(res.content)
                
                # Speech to Text, AI, Text to Speech
                user_text = speech_to_text(input_audio_path, os.getenv("GEMINI_API_KEY"))
                ai_reply = process_text_with_ai(user_text, user_id=sender_id)
                output_audio_path = f"reply_{sender_id}.wav"
                text_to_speech(ai_reply, os.getenv("GEMINI_API_KEY"), output_audio_path)
                
                # Send back via Messenger
                send_audio_reply_messenger(sender_id, output_audio_path)
                
                # Cleanup
                if os.path.exists(input_audio_path): os.remove(input_audio_path)
                if os.path.exists(output_audio_path): os.remove(output_audio_path)
            elif text_body:
                print(f"Received text from Messenger user {sender_id}.")
                ai_reply = process_text_with_ai(text_body, user_id=sender_id)
                send_text_reply_messenger(sender_id, ai_reply)
            else:
                print("Messenger message is not audio or text.")
                
        # ==========================================
        # 2. WHATSAPP WEBHOOK
        # ==========================================
        elif obj == "whatsapp_business_account":
            if not cfg.get("whatsapp_active", True):
                print("WhatsApp is disabled via dashboard.")
                return Response(content="EVENT_RECEIVED", status_code=200)
            changes = entry.get("changes", [])[0]
            value = changes.get("value", {})
            
            if "messages" in value:
                message = value["messages"][0]
                phone_number_id = value["metadata"]["phone_number_id"]
                sender_id = message["from"]
                
                if message.get("type") == "audio":
                    media_id = message["audio"]["id"]
                    print(f"Received audio from WhatsApp user {sender_id}.")
                    
                    input_audio_path = f"incoming_{media_id}.ogg"
                    whatsapp_token = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
                    download_audio_from_meta(media_id, whatsapp_token, input_audio_path)
                    
                    user_text = speech_to_text(input_audio_path, os.getenv("GEMINI_API_KEY"))
                    ai_reply = process_text_with_ai(user_text, user_id=sender_id)
                    
                    output_audio_path = f"reply_{media_id}.wav"
                    text_to_speech(ai_reply, os.getenv("GEMINI_API_KEY"), output_audio_path)
                    
                    send_audio_reply_whatsapp(phone_number_id, sender_id, output_audio_path)
                    
                    if os.path.exists(input_audio_path): os.remove(input_audio_path)
                    if os.path.exists(output_audio_path): os.remove(output_audio_path)
                elif message.get("type") == "text":
                    text_body = message["text"]["body"]
                    print(f"Received text from WhatsApp user {sender_id}.")
                    ai_reply = process_text_with_ai(text_body, user_id=sender_id)
                    send_text_reply_whatsapp(phone_number_id, sender_id, ai_reply)
                else:
                    print("WhatsApp message is not audio or text.")
                    
    except Exception as e:
        print(f"Error processing message: {e}")


def send_text_reply_messenger(recipient_id, text):
    import requests
    import os
    messenger_token = os.getenv("MESSENGER_ACCESS_TOKEN", "")
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={messenger_token}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text}
    }
    print("Sending text to Messenger...")
    send_res = requests.post(url, json=payload)
    if send_res.status_code == 200:
        print("Successfully sent text reply to Messenger user!")
    else:
        print("Failed to send text to Messenger:", send_res.text)

def send_text_reply_whatsapp(phone_number_id, recipient_id, text):
    import requests
    import os
    whatsapp_token = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    send_url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {whatsapp_token}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "type": "text",
        "text": {"body": text}
    }
    print("Sending text to WhatsApp...")
    send_res = requests.post(send_url, headers=headers, json=payload)
    if send_res.status_code == 200:
        print("Successfully sent text reply to WhatsApp user!")
    else:
        print("Failed to send text to WhatsApp:", send_res.text)


def send_audio_reply_messenger(recipient_id, audio_path):
    """Sends audio reply via Facebook Messenger API."""
    import requests
    messenger_token = os.getenv("MESSENGER_ACCESS_TOKEN", "")
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={messenger_token}"
    payload = {
        "recipient": json.dumps({"id": recipient_id}),
        "message": json.dumps({"attachment": {"type": "audio", "payload": {"is_reusable": True}}})
    }
    print("Uploading and sending audio to Messenger...")
    with open(audio_path, "rb") as f:
        files = {"filedata": (audio_path, f, "audio/mpeg")}
        send_res = requests.post(url, data=payload, files=files)
        if send_res.status_code == 200:
            print("Successfully sent voice reply to Messenger user!")
        else:
            print("Failed to send to Messenger:", send_res.text)

def send_audio_reply_whatsapp(phone_number_id, recipient_id, audio_path):
    """Uploads the generated audio and sends it via WhatsApp Business API."""
    import requests
    whatsapp_token = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    
    # URL for uploading media
    upload_url = f"https://graph.facebook.com/v18.0/{phone_number_id}/media"
    headers = {"Authorization": f"Bearer {whatsapp_token}"}
    
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





