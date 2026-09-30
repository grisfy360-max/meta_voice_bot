import json
def get_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from duckduckgo_search import DDGS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

from langchain_core.callbacks import BaseCallbackHandler

# কাস্টম লগার তৈরি করা, যা ফেইল করলে প্রিন্ট করবে
class FallbackLogger(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs):
        print(f"\n[⚠️ LLM Routing Alert] Error encountered: {error}")
        print("[🔄 Action] Shifting to the next smaller fallback model...\n")

fallback_logger = FallbackLogger()

# ১. Initialize LLMs (প্রাইমারি এবং ফলব্যাক)
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    google_api_key=get_config().get("gemini_api_key", os.getenv("GEMINI_API_KEY")),
    temperature=0.7,
    max_retries=0,
    callbacks=[fallback_logger]
)

fallback_llm_1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=get_config().get("gemini_api_key", os.getenv("GEMINI_API_KEY")),
    temperature=0.7,
    max_retries=0,
    callbacks=[fallback_logger]
)

fallback_llm_2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=get_config().get("gemini_api_key", os.getenv("GEMINI_API_KEY")),
    temperature=0.7,
    max_retries=0,
    callbacks=[fallback_logger]
)

# LLM Routing Setup (প্রাইমারি ফেইল করলে ছোট মডেলে ফলব্যাক করবে)
llm = primary_llm.with_fallbacks([fallback_llm_1, fallback_llm_2])

# ২. Initialize Embeddings (টেক্সটকে ভেক্টরে রূপান্তর করার জন্য)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=get_config().get("gemini_api_key", os.getenv("GEMINI_API_KEY"))
)

# ৩. Initialize ChromaDB (Vector Database)
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ৪. Prompt Template (বটকে ইনস্ট্রাকশন দেওয়া)
system_prompt = (
    "You are a highly intelligent and friendly AI Voice Assistant for Meta platforms (WhatsApp/Messenger). "
    "CRITICAL RULE: You MUST always reply ONLY in pure Bengali script (বাংলা অক্ষরে). "
    "CULTURAL RULE: If you are greeting the user for the very first time in a conversation, use 'আসসালামু আলাইকুম' (Assalamu Alaikum) or 'হ্যালো' (Hello). DO NOT use 'নমস্কার'. However, if the conversation is already ongoing (you have chatted recently), DO NOT repeat the greeting. Start answering directly. "
    "IMAGE RULE: If the user asks about a picture or image (e.g. 'what is this picture') but no image is visible in the context, DO NOT ask them to upload it or say you can't see it. Instead, say 'আমি আপনার ছবিটি দেখছি, একটু অপেক্ষা করুন।' (I am looking at your picture, please wait a moment). "
    "Keep answers conversational, friendly, and short (1-2 sentences), as it will be converted to a voice note. "
    "Use the retrieved context to answer the user's question if relevant. If you don't know, politely say you don't know.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"), # মেমোরি
    ("human", "{input}"), # ইউজারের ইনপুট
])

# ৫. Create LCEL Chains
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.runnables import RunnableLambda

def custom_output_parser(ai_message):
    content = ai_message.content
    if isinstance(content, list):
        # Extract the actual text from the list of dicts (Gemini 3.5 format bug)
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
        return " ".join(texts)
    return str(content)

rag_chain = (
    {"context": itemgetter("input") | retriever | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
    | prompt
    | llm
    | RunnableLambda(custom_output_parser)
)

# ৬. Memory (চ্যাট হিস্ট্রি মনে রাখার জন্য)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# RAG এবং Memory কে একসাথে চেইন করা
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Search Decision Chain
search_decision_prompt = ChatPromptTemplate.from_template("""
You are a smart router. Decide if you need to search the live internet to answer the user's question (e.g., current news, latest prices, weather, facts not in your knowledge base).
If YES, reply ONLY with: SEARCH: <your search query>
If NO (can be answered normally or from chat history), reply ONLY with: NO

User question: {input}
""")
search_decision_chain = search_decision_prompt | llm | RunnableLambda(custom_output_parser)

def process_text_with_ai(user_input: str, user_id: str = "default_user"):
    """
    এই ফাংশনটি ইউজারের প্রশ্ন নেবে, ডেটাবেস খুঁজবে, ইন্টারনেট সার্চ করবে (যদি প্রয়োজন হয়), 
    আগের কথা মনে করবে এবং একটি সুন্দর বাংলা উত্তর (Text) তৈরি করে ফেরত দেবে।
    """
    try:
        # Step 1: Decide if we need internet search
        decision = search_decision_chain.invoke({"input": user_input})
        internet_context = ""
        
        if decision.strip().startswith("SEARCH:"):
            query = decision.replace("SEARCH:", "").strip()
            print(f"Searching internet for: {query}")
            try:
                results = DDGS().text(query, max_results=3)
                if results:
                    snippets = [f"Title: {r['title']}\nSnippet: {r['body']}" for r in results]
                    internet_context = "INTERNET SEARCH RESULTS:\n" + "\n\n".join(snippets) + "\n\n"
            except Exception as e:
                print(f"DDG Search error: {e}")
        
        # Step 2: Combine inputs
        final_input = user_input
        if internet_context:
            final_input = f"{internet_context}\nUser Question: {user_input}"
            
        # Step 3: Run RAG chain
        response = conversational_rag_chain.invoke(
            {"input": final_input},
            config={"configurable": {"session_id": user_id}}
        )
        return response
    except Exception as e:
        print(f"Error in LangChain agent: {e}")
        return "দুঃখিত, আমি আপনার কথাটি ঠিক বুঝতে পারিনি। আরেকবার বলবেন কি?"



# Image Processing Chain (No RAG needed for raw images)
image_system_prompt = (
    "You are a highly intelligent and friendly AI Voice Assistant. "
    "CRITICAL RULE: You MUST always reply ONLY in pure Bengali script (বাংলা অক্ষরে). "
    "CULTURAL RULE: If you are greeting the user for the very first time in a conversation, use 'আসসালামু আলাইকুম' (Assalamu Alaikum) or 'হ্যালো' (Hello). DO NOT use 'নমস্কার'. However, if the conversation is already ongoing (you have chatted recently), DO NOT repeat the greeting. Start answering directly. "
    "Keep answers conversational, friendly, and short (1-2 sentences). "
    "Describe the image the user uploaded or answer their question about it."
)

image_prompt = ChatPromptTemplate.from_messages([
    ("system", image_system_prompt),
    MessagesPlaceholder("chat_history"),
    MessagesPlaceholder("input"),
])

image_chain = image_prompt | llm | RunnableLambda(custom_output_parser)

conversational_image_chain = RunnableWithMessageHistory(
    image_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def process_image_with_ai(image_url: str, user_id: str = "default_user", custom_prompt: str = None):
    try:
        import requests, base64, os
        # Download securely. Facebook scontent URLs are pre-signed, so Authorization header can sometimes cause CDN blocks (returning a 1x1 or placeholder image).
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        params = {
            "access_token": get_config().get("messenger_token", os.getenv("PAGE_ACCESS_TOKEN", ""))
        }
        print(f"DEBUG: Downloading image URL: {image_url}")
        img_response = requests.get(image_url, headers=headers, params=params)
        
        content_type = img_response.headers.get("Content-Type", "")
        print(f"DEBUG: Downloaded Content-Type: {content_type}, Status: {img_response.status_code}")
        
        if "image" not in content_type:
            return "দুঃখিত, ছবিটি আমার কাছে ঠিকমতো পৌঁছায়নি। ফেসবুক সিকিউরিটি ব্লক করেছে।"
            
        img_b64 = base64.b64encode(img_response.content).decode("utf-8")
        mime_type = content_type if content_type else "image/jpeg"
        
        # Use custom prompt if provided, else fallback to default
        prompt_text = custom_prompt if custom_prompt else "Can you describe this image or answer what it is? Always reply in pure Bengali."
        
        from langchain_core.messages import HumanMessage
        image_message = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}}
        ]
        
        response = conversational_image_chain.invoke(
            {"input": [HumanMessage(content=image_message)]},
            config={"configurable": {"session_id": user_id}}
        )
        return response
    except Exception as e:
        print(f"Error in LangChain image agent: {e}")
        return "দুঃখিত, আমি ছবিটি ঠিকমতো দেখতে পাচ্ছি না।"
