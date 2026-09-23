import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# ১. Initialize LLM (ব্রেইন)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

# ২. Initialize Embeddings (টেক্সটকে ভেক্টরে রূপান্তর করার জন্য)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY")
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

rag_chain = (
    {"context": itemgetter("input") | retriever | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
    | prompt
    | llm
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
    output_messages_key="answer",
)

def process_text_with_ai(user_input: str, user_id: str = "default_user"):
    """
    এই ফাংশনটি ইউজারের প্রশ্ন নেবে, ডেটাবেস খুঁজবে, আগের কথা মনে করবে 
    এবং একটি সুন্দর বাংলা উত্তর (Text) তৈরি করে ফেরত দেবে।
    """
    try:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": user_id}}
        )
        return response["answer"]
    except Exception as e:
        print(f"Error in LangChain agent: {e}")
        return "দুঃখিত, আমি আপনার কথাটি ঠিক বুঝতে পারিনি। আরেকবার বলবেন কি?"
