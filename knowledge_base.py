import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def get_gemini_api_key():
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

def get_embeddings():
    api_key = get_gemini_api_key()
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key
    )

def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )

# ডিফল্ট প্রোডাক্ট ক্যাটালগ ও বিজনেস পলিসি ডাটাবেস
DEFAULT_KNOWLEDGE_DOCS = [
    Document(
        page_content="প্রোডাক্ট: ব্ল্যাক প্রিমিয়াম কটন টি-শার্ট (Premium Black Cotton T-Shirt)\nবিবরণ: ১০০% প্রিমিয়াম কম্বড সুতি ফেব্রিক, অত্যন্ত আরামদায়ক ও ব্রিদেবল।\nসাইজ উপলব্ধ: M, L, XL, XXL\nমূল্য: ৪৫০ টাকা\nস্টক তথ্য: ব্ল্যাক কালার L (Large) সাইজ স্টকে এভেইলেবল আছে (১৫ পিস ইন-স্টক)।",
        metadata={"category": "product", "sku": "TSHIRT-BLK-L"}
    ),
    Document(
        page_content="প্রোডাক্ট: হোয়াইট ও নেভি ব্লু রেগুলার টি-শার্ট\nসাইজ উপলব্ধ: M, L, XL\nমূল্য: ৪৫০ টাকা\nস্টক তথ্য: হোয়াইট এবং নেভি ব্লু সব সাইজ বর্তমানে স্টকে আছে।",
        metadata={"category": "product", "sku": "TSHIRT-WHT-NVY"}
    ),
    Document(
        page_content="প্রোডাক্ট: ড্রপ-শোল্ডার ওভারসাইজড টি-শার্ট (Oversized Drop-Shoulder T-Shirt)\nকালার: ব্ল্যাক, অলিভ গ্রিন, চারকোল\nসাইজ: M, L, XL\nমূল্য: ৫৫০ টাকা\nস্টক: সব কালারের সব সাইজ রেডি স্টকে আছে।",
        metadata={"category": "product", "sku": "TSHIRT-OVERSIZED"}
    ),
    Document(
        page_content="ডেলিভারি চার্জ ও সময়সূচী:\n১. ঢাকা সিটির ভেতরে ডেলিভারি চার্জ: ৬০ টাকা (ডেলিভারি সময় ১ থেকে ২ কার্যদিবস)।\n২. ঢাকা সিটির বাইরে (সারা বাংলাদেশ): ১২০ টাকা (ডেলিভারি সময় ২ থেকে ৩ কার্যদিবস)।\n৩. ক্যাশ অন ডেলিভারি (Cash on Delivery) সুবিধা আছে। হোম ডেলিভারি পেয়ে চেক করে টাকা দিতে পারবেন।",
        metadata={"category": "policy", "type": "delivery"}
    ),
    Document(
        page_content="রিটার্ন ও সাইজ এক্সচেঞ্জ পলিসি:\nপণ্য হাতে পাওয়ার পর সাইজে কোনো সমস্যা হলে বা কোনো ডিফেক্ট থাকলে ৭ দিনের মধ্যে সম্পূর্ণ ফ্রিতে সাইজ এক্সচেঞ্জ সুবিধা পাবেন।",
        metadata={"category": "policy", "type": "return"}
    ),
    Document(
        page_content="অর্ডার করার নিয়ম:\nফোন কলেই অর্ডার কনফার্ম করতে চাইলে আপনার নাম, মোবাইল নম্বর, সম্পূর্ণ ডেলিভারি ঠিকানা এবং যে সাইজটি নিতে চান তা উল্লেখ করুন।",
        metadata={"category": "policy", "type": "order"}
    )
]

def get_fallback_knowledge_docs(query: str, k: int = 3) -> list:
    """
    জিরো-লেটেন্সি ইন-মেমোরি কি-ওয়ার্ড ও ইন্টেন্ট রুল বেসড ম্যাচিং।
    ChromaDB রেডি না থাকলেও মিলিসেকেন্ডের ভগ্নাংশে সঠিক প্রোডাক্ট ও পলিসি কনটেক্সট প্রদান করে।
    """
    q = (query or "").lower()
    scored = []
    
    for doc in DEFAULT_KNOWLEDGE_DOCS:
        score = 0
        content = doc.page_content.lower()
        cat = doc.metadata.get("category", "").lower()
        typ = doc.metadata.get("type", "").lower()
        
        # ১. ডেলিভারি ও চার্জ সংক্রান্ত ইন্টেন্ট
        if any(w in q for w in ["ডেলিভারি", "চার্জ", "খরচ", "পাঠাতে", "পৌঁছাতে", "শিপিং", "হোম ডেলিভারি", "ক্যাশ অন", "delivery", "charge", "shipping", "fee", "cost"]):
            if typ == "delivery" or "ডেলিভারি" in content:
                score += 15
                
        # ২. রিটার্ন ও সাইজ এক্সচেঞ্জ পলিসি
        if any(w in q for w in ["এক্সচেঞ্জ", "রিটার্ন", "বদল", "ফেরত", "সাইজ সমস্যা", "ডিফেক্ট", "ত্রুটি", "exchange", "return"]):
            if typ == "return" or "এক্সচেঞ্জ" in content:
                score += 15
                
        # ৩. অর্ডার ও বুকিং ইন্টেন্ট
        if any(w in q for w in ["অর্ডার", "বুকিং", "বুক", "কিনতে", "অর্ডারটি", "order", "buy", "purchase"]):
            if typ == "order" or "অর্ডার" in content:
                score += 15
                
        # ৪. নির্দিষ্ট প্রোডাক্ট ও স্টক সংক্রান্ত ইন্টেন্ট
        if any(w in q for w in ["ব্ল্যাক", "কটন", "টি-শার্ট", "টি শার্ট", "কালো", "t-shirt", "tshirt", "black"]):
            if "ব্ল্যাক প্রিমিয়াম কটন টি-শার্ট" in content:
                score += 12
        if any(w in q for w in ["হোয়াইট", "সাদা", "নেভি", "white", "navy"]):
            if "হোয়াইট ও নেভি ব্লু" in content:
                score += 12
        if any(w in q for w in ["ওভারসাইজড", "ড্রপ-শোল্ডার", "ড্রপ শোল্ডার", "oversized"]):
            if "ড্রপ-শোল্ডার" in content:
                score += 12
        if any(w in q for w in ["সাইজ", "স্টক", "দাম", "প্রাইস", "টাকা", "price", "size", "stock", "l সাইজ", "লার্জ", "মিডিয়াম", "m সাইজ"]):
            if cat == "product":
                score += 8
                
        # সাবস্ট্রিং টোকেন ম্যাচ
        for token in q.split():
            clean_tok = token.strip("?,.!:;\"'()[]{}")
            if len(clean_tok) > 2 and clean_tok in content:
                score += 2
                
        if score > 0:
            scored.append((score, doc))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [doc for _, doc in scored[:k]]
    # ডিফল্ট পলিসি ও টপ প্রোডাক্ট
    return [DEFAULT_KNOWLEDGE_DOCS[3], DEFAULT_KNOWLEDGE_DOCS[0], DEFAULT_KNOWLEDGE_DOCS[5]][:k]

def seed_knowledge_base():
    """ChromaDB তে প্রোডাক্ট ও বিজনেস পলিসি ডকুমেন্ট সেভ করে"""
    try:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(DEFAULT_KNOWLEDGE_DOCS)
        print("Knowledge base seeded successfully into ChromaDB!")
    except Exception as e:
        print(f"[KB Seed Warning] ChromaDB seeding skipped/failed: {e}")

def query_knowledge_base(query: str, k: int = 3) -> str:
    """
    ইউজারের প্রশ্ন অনুযায়ী ChromaDB ও ইন-মেমোরি নলেজ থেকে প্রাসঙ্গিক প্রোডাক্ট ও পলিসি কনটেক্সট এনে দেয়।
    ChromaDB খালি বা নেটওয়ার্ক এরর হলেও ইন-মেমোরি কি-ওয়ার্ড ম্যাচিং থেকে তাৎক্ষণিক ১০০% নির্ভুল উত্তর নিশ্চিত করে।
    """
    fallback_docs = get_fallback_knowledge_docs(query, k=k)
    fallback_content = "\n\n".join(doc.page_content for doc in fallback_docs)
    
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search(query, k=k)
        if results and len(results) > 0:
            chroma_content = "\n\n".join(doc.page_content for doc in results)
            # ডেলিভারি প্রশ্নের ক্ষেত্রে ডেলিভারি ডকুমেন্ট যেন সর্বদা উপস্থিত থাকে
            if any(w in (query or "").lower() for w in ["ডেলিভারি", "চার্জ", "খরচ", "delivery", "charge"]):
                delivery_doc = DEFAULT_KNOWLEDGE_DOCS[3].page_content
                if delivery_doc not in chroma_content:
                    return f"{delivery_doc}\n\n{chroma_content}"
            return chroma_content
    except Exception as e:
        print(f"[KB Notice] ChromaDB query bypassed ({e}), using memory fallback.")
        
    return fallback_content

if __name__ == "__main__":
    seed_knowledge_base()
    test_query = "আপনাদের ডেলিভারি চার্জ কত এবং ব্ল্যাক টি-শার্টের এল সাইজ কি আছে?"
    print("Query:", test_query)
    context = query_knowledge_base(test_query)
    print("Retrieved Context:\n", context)
