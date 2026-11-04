from fastapi.staticfiles import StaticFiles

import os

from fastapi import FastAPI, Request, Response, BackgroundTasks

from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv



load_dotenv()



app = FastAPI(title="Meta Voice Assistant API")

app.mount("/static", StaticFiles(directory="static"), name="static")



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



import json
from fastapi.responses import HTMLResponse, FileResponse

def get_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

VERIFY_TOKEN = get_config().get("verify_token", "my_verify_token_123")
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")





import asyncio

import httpx



async def keep_alive():

    while True:

        try:

            async with httpx.AsyncClient() as client:

                await client.get("https://meta-voice-bot.onrender.com/")

                print("Pinged self to stay awake")

        except Exception as e:

            print("Self-ping failed:", e)

        await asyncio.sleep(840)  # Ping every 14 minutes (840 seconds)



@app.on_event("startup")

async def startup_event():

    asyncio.create_task(keep_alive())



@app.get("/", response_class=HTMLResponse)

def dashboard():
    return FileResponse("dashboard.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

# Global debug logs
debug_logs = []

@app.get("/debug")
def get_debug_logs():
    return {"logs": debug_logs[-10:]}



@app.get("/api/config")

def read_config():

    return get_config()



from pydantic import BaseModel

class PreviewRequest(BaseModel):

    voice: str

    persona: str



@app.post("/api/preview_voice")

def preview_voice_post(req: PreviewRequest):

    import os

    import time

    from audio_services import text_to_speech

    

    api_key = get_config().get("gemini_api_key", "")

    if not api_key:

        return Response(content="Missing API Key", status_code=500)

        

    text = "à¦¹à§à¦¯à¦¾à¦²à§‹! à¦†à¦®à¦¿ à¦†à¦ªà¦¨à¦¾à¦° à¦à¦†à¦‡ à¦…à§à¦¯à¦¾à¦¸à¦¿à¦¸à§à¦Ÿà§à¦¯à¦¾à¦¨à§à¦Ÿà¥¤ à¦†à¦®à¦¾à¦° à¦­à§Ÿà§‡à¦¸ à¦ à¦¿à¦• à¦à¦°à¦•à¦® à¦¶à§‹à¦¨à¦¾à¦¬à§‡à¥¤"

    file_path = f"preview_{req.voice}_{int(time.time())}.wav"

    

    try:

        audio_path = text_to_speech(text, api_key, file_path, override_voice=req.voice, override_persona=req.persona)

        return FileResponse(audio_path, media_type="audio/wav")

    except Exception as e:

        return Response(content=str(e), status_code=500)



@app.post("/api/config")

async def update_config(request: Request):

    data = await request.json()

    with open("config.json", "w", encoding="utf-8") as f:

        json.dump(data, f, indent=4)

    return {"status": "success"}

@app.get("/api/knowledge")
def get_knowledge_api():
    try:
        from knowledge_base import DEFAULT_KNOWLEDGE_DOCS
        items = [{"content": d.page_content, "metadata": d.metadata} for d in DEFAULT_KNOWLEDGE_DOCS]
        return {"status": "success", "items": items}
    except Exception as e:
        return {"status": "error", "message": str(e), "items": []}

@app.post("/api/knowledge/sync")
def sync_knowledge_api():
    try:
        from knowledge_base import seed_knowledge_base
        seed_knowledge_base()
        return {"status": "success", "message": "Knowledge base synced to ChromaDB successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 1. Webhook Verification (Meta will call this during setup)

@app.get("/webhook")

async def verify_webhook(request: Request):

    mode = request.query_params.get("hub.mode")

    token = request.query_params.get("hub.verify_token")

    challenge = request.query_params.get("hub.challenge")

    

    cfg = get_config()

    current_verify_token = cfg.get("verify_token", VERIFY_TOKEN)



    if mode and token:

        if mode == "subscribe" and token == current_verify_token:

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

        cfg = get_config()

    except Exception:

        cfg = {}

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
            
            # Instantly send typing indicator
            send_typing_on_messenger(sender_id)


            

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

                user_text = speech_to_text(input_audio_path, get_config().get("gemini_api_key", ""))

                ai_reply = process_text_with_ai(user_text, user_id=sender_id)
                send_typing_on_messenger(sender_id) # Refresh typing indicator
                
                                
                output_audio_path = f"reply_{sender_id}.wav"

                text_to_speech(ai_reply, get_config().get("gemini_api_key", ""), output_audio_path)

                

                # Send back via Messenger

                send_audio_reply_messenger(sender_id, output_audio_path)

                

                # Cleanup

                if os.path.exists(input_audio_path): os.remove(input_audio_path)

                if os.path.exists(output_audio_path): os.remove(output_audio_path)

            elif attachments and attachments[0]["type"] == "image":
                image_url = attachments[0]["payload"]["url"]
                print(f"Received image from Messenger user {sender_id}.")
                from langchain_agent import process_image_with_ai
                ai_reply = process_image_with_ai(image_url, user_id=sender_id, custom_prompt=text_body)
                send_typing_on_messenger(sender_id) # Refresh typing indicator
                send_text_reply_messenger(sender_id, ai_reply)

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

                    whatsapp_token = get_config().get("whatsapp_token", "")

                    download_audio_from_meta(media_id, whatsapp_token, input_audio_path)

                    

                    user_text = speech_to_text(input_audio_path, get_config().get("gemini_api_key", ""))

                    ai_reply = process_text_with_ai(user_text, user_id=sender_id)
                    send_typing_on_messenger(sender_id) # Refresh typing indicator

                    

                    output_audio_path = f"reply_{media_id}.wav"

                    text_to_speech(ai_reply, get_config().get("gemini_api_key", ""), output_audio_path)

                    

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






def send_typing_on_messenger(recipient_id):
    import requests
    import os
    messenger_token = get_config().get("messenger_token", "")
    if not messenger_token: return
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={messenger_token}"
    payload = {
        "recipient": {"id": recipient_id},
        "sender_action": "typing_on"
    }
    requests.post(url, json=payload)

def send_text_reply_messenger(recipient_id, text):

    import requests

    import os

    messenger_token = get_config().get("messenger_token", "")

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

    whatsapp_token = get_config().get("whatsapp_token", "")

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

    messenger_token = get_config().get("messenger_token", "")

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

    whatsapp_token = get_config().get("whatsapp_token", "")

    

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
        global debug_logs
        debug_logs.append({"webhook": data})
        if len(debug_logs) > 50: debug_logs.pop(0)
        
        # We send the processing to a background task so Meta gets a fast 200 OK
        background_tasks.add_task(process_voice_message, data)

        

    except Exception as e:

        print(f"Error parsing webhook data: {e}")



    return Response(content="EVENT_RECEIVED", status_code=200)


# ============================================================================
# Live Voice Call AI Engine (STT -> RAG -> Gemini Flash -> Streaming Edge-TTS)
# ============================================================================
from fastapi import WebSocket, WebSocketDisconnect
import base64
import uuid
import re
from typing import Optional
from google import genai
from audio_services import transcribe_audio_bytes, stream_edge_tts, get_gemini_api_key
from knowledge_base import query_knowledge_base

@app.get("/call", response_class=HTMLResponse)
def live_call_page():
    """Serves the real-time voice call client screen."""
    file_path = os.path.join(os.path.dirname(__file__), "live_call.html")
    if not os.path.exists(file_path):
        return HTMLResponse("<h3>Error: live_call.html not found.</h3>", status_code=404)
    return FileResponse(
        file_path, 
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

BENGALI_VOICE_SYSTEM_PROMPT = """You are a warm, polite, and natural conversational Voice AI Customer Support Representative for an e-commerce fashion shop in Bangladesh.
You are speaking directly with a customer over a live phone call in natural spoken conversational Bengali (বাংলা কথ্য রূপ).

CRITICAL VOICE CALL RULES:
1. CONVERSATIONAL BREVITY: Keep every response short and concise (strictly 1 to 3 spoken sentences maximum). The customer is listening to audio over the phone, not reading text.
2. ZERO MARKDOWN / FORMATTING: Absolutely NEVER use markdown formatting, asterisks (**), bullet points (*, -), hashes (#), or numbered lists (1., 2.). TTS synthesizers will speak punctuation aloud if you include markdown. Write pure flowing spoken text only.
3. CULTURAL GREETINGS:
   - On the very first greeting of the call, greet politely with 'আসসালামু আলাইকুম' or 'হ্যালো'.
   - Once the call is ongoing, NEVER repeat greetings. Answer directly and naturally.
4. KNOWLEDGE GROUNDING:
   - Base all answers strictly on the provided Store Knowledge Context below.
   - Premium Black Cotton T-Shirt L size is in stock (15 pcs available), price is 450 BDT.
   - Delivery charges: 60 BDT inside Dhaka (1-2 days), 120 BDT outside Dhaka (2-3 days). Cash on delivery available.
   - Size exchange policy: 7 days free size exchange.
   - When a customer expresses interest or asks to buy/order, politely invite them: 'অর্ডারটি কনফার্ম করতে আপনার নাম, মোবাইল নম্বর এবং সম্পূর্ণ ডেলিভারি ঠিকানাটি দিন।'
5. UNKNOWN QUESTIONS: If a question is not covered in the context, politely apologize in sweet Bengali and offer to connect them with a human supervisor.

Store Knowledge Context:
{context}"""

def clean_voice_text(text: str) -> str:
    """
    Sanitizes generated text to guarantee zero TTS-spoken punctuation.
    Preserves Bengali words with internal hyphens (e.g. 'টি-শার্ট')
    while stripping all markdown formatting, asterisks, hashes, and bullets.
    """
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(\d+|[০-৯]+)\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[`~]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

async def generate_voice_response(
    query: str, 
    chat_history: list, 
    session_id: str
) -> tuple[str, str]:
    """
    Queries ChromaDB RAG, injects grounded store facts, and returns (response_text, retrieved_context).
    """
    api_key = get_gemini_api_key()
    client = genai.Client(api_key=api_key)
    
    # Retrieve ChromaDB context (k=3 captures compound queries)
    try:
        context = await asyncio.to_thread(query_knowledge_base, query, 3)
    except Exception as e:
        print(f"[RAG Error] {e}")
        context = ""
        
    formatted_context = context.strip() if context and context.strip() else "প্রাসঙ্গিক কোনো স্টোর তথ্য পাওয়া যায়নি।"
    system_instruction = BENGALI_VOICE_SYSTEM_PROMPT.format(context=formatted_context)
    
    # Format message history for Gemini
    messages = [f"System: {system_instruction}"]
    for turn in chat_history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        messages.append(f"{role}: {turn.get('text', '')}")
    messages.append(f"User: {query}")
    messages.append("Assistant:")
    
    full_prompt = "\n\n".join(messages)
    
    candidate_models = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-2.5-flash"]
    for model_name in candidate_models:
        for attempt in range(2):
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=full_prompt
                )
                if response and response.text and response.text.strip():
                    cleaned = clean_voice_text(response.text.strip())
                    return cleaned, formatted_context
            except Exception as e:
                print(f"[LLM Warning] Model {model_name} attempt {attempt+1} failed: {e}")
                await asyncio.sleep(0.3)
                
    fallback_text = "জি, আমি আপনার বিষয়টি বুঝতে পেরেছি। আপনাকে কিভাবে আরও সহায়তা করতে পারি?"
    return fallback_text, formatted_context

class CallSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_buffer = bytearray()
        self.active_task: Optional[asyncio.Task] = None
        self.chat_history: list = []
        self.audio_format: str = "binary"  # 'binary' or 'base64'

@app.websocket("/ws/live-call")
async def websocket_live_call_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = CallSession(session_id)
    print(f"[WS Connect] Live call session started: {session_id}")
    
    # Initial status
    await websocket.send_json({"type": "status", "status": "listening", "session_id": session_id})
    
    async def cancel_active_turn():
        """Cancels ongoing STT/LLM/TTS operations immediately upon user barge-in."""
        if session.active_task and not session.active_task.done():
            print(f"[Barge-In] Cancelling active turn for session {session_id}")
            session.active_task.cancel()
            try:
                await session.active_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[Barge-In] Exception in cancelled task: {e}")
            session.active_task = None
            await websocket.send_json({"type": "interrupted"})
            await websocket.send_json({"type": "status", "status": "listening"})

    async def run_turn(audio_payload: bytes):
        """Processes user audio -> STT -> RAG -> Gemini -> Streaming Edge-TTS."""
        try:
            await websocket.send_json({"type": "status", "status": "processing"})
            
            # Step 1: Zero-disk inline STT
            mime_type = "audio/webm"
            if audio_payload.startswith(b"RIFF"):
                mime_type = "audio/wav"
            elif audio_payload.startswith(b"ID3") or (len(audio_payload) >= 2 and audio_payload[0] == 0xFF):
                mime_type = "audio/mp3"
                
            transcript = await transcribe_audio_bytes(audio_payload, mime_type=mime_type)
            if not transcript:
                print("[WS Warning] Empty transcript. Resetting to listening.")
                await websocket.send_json({"type": "status", "status": "listening"})
                return
                
            await websocket.send_json({"type": "transcript", "text": transcript})
            session.chat_history.append({"role": "user", "text": transcript})
            
            # Step 2: Grounded RAG & LLM response
            response_text, rag_context = await generate_voice_response(
                query=transcript,
                chat_history=session.chat_history,
                session_id=session.session_id
            )
            
            await websocket.send_json({"type": "rag_context", "context": rag_context})
            await websocket.send_json({"type": "llm_response", "text": response_text})
            session.chat_history.append({"role": "bot", "text": response_text})
            
            # Step 3: Stream synthesized speech
            await websocket.send_json({"type": "status", "status": "speaking"})
            
            async for audio_chunk in stream_edge_tts(response_text, voice="bn-BD-NabanitaNeural"):
                if session.audio_format == "base64":
                    b64_data = base64.b64encode(audio_chunk).decode("utf-8")
                    await websocket.send_json({"type": "audio_chunk", "data": b64_data})
                else:
                    await websocket.send_bytes(audio_chunk)
                await asyncio.sleep(0.001)
                
            await websocket.send_json({"type": "audio_end"})
            await websocket.send_json({"type": "status", "status": "listening"})
            
        except asyncio.CancelledError:
            print(f"[WS Task Cancelled] Turn task aborted: {session_id}")
            raise
        except (WebSocketDisconnect, RuntimeError):
            print(f"[WS Disconnect] Connection closed during turn: {session_id}")
            return
        except Exception as e:
            print(f"[WS Error] Turn execution failed: {e}")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
                await websocket.send_json({"type": "status", "status": "listening"})
            except Exception:
                pass

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            
            # Direct binary audio chunk
            if "bytes" in message and message["bytes"]:
                if session.active_task and not session.active_task.done():
                    await cancel_active_turn()
                session.audio_buffer.extend(message["bytes"])
                continue
                
            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    continue
                    
                msg_type = payload.get("type")
                
                if msg_type == "init":
                    if payload.get("session_id"):
                        session.session_id = payload["session_id"]
                    if payload.get("audio_format"):
                        session.audio_format = payload["audio_format"]
                    elif payload.get("format"):
                        session.audio_format = payload["format"]
                    session.audio_buffer.clear()
                    await websocket.send_json({"type": "status", "status": "listening", "session_id": session.session_id})
                    
                elif msg_type == "audio_chunk":
                    if session.active_task and not session.active_task.done():
                        await cancel_active_turn()
                    b64_data = payload.get("data", "")
                    if b64_data:
                        session.audio_buffer.extend(base64.b64decode(b64_data))
                        
                elif msg_type == "barge_in":
                    session.audio_buffer.clear()
                    await cancel_active_turn()
                    
                elif msg_type in ["audio_end", "user_speech_end"]:
                    if len(session.audio_buffer) > 0:
                        raw_audio = bytes(session.audio_buffer)
                        session.audio_buffer.clear()
                        await cancel_active_turn()
                        session.active_task = asyncio.create_task(run_turn(raw_audio))
                    else:
                        await websocket.send_json({"type": "status", "status": "listening"})
                        
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    
                elif msg_type == "end_call":
                    await cancel_active_turn()
                    break

    except WebSocketDisconnect:
        print(f"[WS Disconnect] Live call disconnected: {session_id}")
    except Exception as e:
        print(f"[WS Error] Exception in socket: {e}")
    finally:
        if session.active_task and not session.active_task.done():
            session.active_task.cancel()




if __name__ == "__main__":

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)























