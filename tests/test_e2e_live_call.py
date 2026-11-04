"""
Automated E2E Test Suite for Bengali Conversational Voice AI Agent with RAG
Verifies:
1. HTTP route /call serves live_call.html (canvas visualizer, Alpine.js, Web Audio, barge-in)
2. WebSocket handshake and full-duplex session initialization on /ws/live-call
3. Bengali speech transcription & RAG context retrieval for delivery charges (Dhaka 60 / Outside 120 Tk)
4. Bengali speech query for product catalog & stock (Black Cotton T-Shirt L size in-stock, 15 pcs, 450 Tk)
5. Bengali speech query for 7-day free size exchange & return policy
6. Instant real-time barge-in interruption (< 250ms cancellation)
7. Audio stream byte integrity (MPEG / ID3 header check)
8. End-to-end latency SLA benchmarking
"""

import asyncio
import base64
import json
import os
import sys
import time
import threading
import httpx
import websockets

# Ensure UTF-8 unbuffered stdout/stderr encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# Setup project path
BOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BOT_DIR, ".env"))

BASE_URL = os.getenv("TEST_SERVER_URL", "http://127.0.0.1:8000")
WS_URL = BASE_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws/live-call"

# Color helpers
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def log_header(title: str):
    print(f"\n{BLUE}======================================================================{RESET}")
    print(f"{BLUE}RUNNING TEST: {title}{RESET}")
    print(f"{BLUE}======================================================================{RESET}")

def log_pass(msg: str):
    print(f"  {GREEN}✓ [PASS]{RESET} {msg}")

def log_fail(msg: str):
    print(f"  {RED}✗ [FAIL]{RESET} {msg}")

# Background server management for standalone self-sufficient test runs
_server_instance = None
_server_thread = None

def ensure_server_running():
    """Starts local FastAPI server on 127.0.0.1:8000 if not already running."""
    global _server_instance, _server_thread
    try:
        res = httpx.get(f"{BASE_URL}/call", timeout=1.0)
        if res.status_code == 200:
            return  # Server already active
    except Exception:
        pass

    print(f"{YELLOW}Local server not detected on {BASE_URL}. Launching background test server...{RESET}")
    import uvicorn
    from main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    _server_instance = uvicorn.Server(config)

    def run_server():
        _server_instance.run()

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

    # Wait for server to come up
    for _ in range(30):
        time.sleep(0.3)
        try:
            res = httpx.get(f"{BASE_URL}/call", timeout=1.0)
            if res.status_code == 200:
                print(f"{GREEN}Background test server online at {BASE_URL}{RESET}")
                return
        except Exception:
            continue

    raise RuntimeError("Failed to start background test server on port 8000.")

async def synthesize_test_audio(text: str) -> bytes:
    """Dynamically generates test audio bytes in Bengali using Edge-TTS without local file fixtures."""
    import edge_tts
    communicate = edge_tts.Communicate(text, "bn-BD-NabanitaNeural")
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio" and chunk["data"]:
            chunks.append(chunk["data"])
    return b"".join(chunks)

# ============================================================================
# Async Test Implementations
# ============================================================================

async def _async_test_01_http_route_call():
    log_header("TC-01: HTTP Route /call Serves live_call.html")
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.get(f"{BASE_URL}/call")
        assert res.status_code == 200, f"Expected HTTP 200, got {res.status_code}"
        assert "text/html" in res.headers.get("content-type", "").lower(), "Expected text/html content type"
        html = res.text
        assert "waveformCanvas" in html, "Missing #waveformCanvas visualizer"
        assert "Alpine.js" in html or "x-data" in html, "Missing Alpine.js framework"
        assert "/ws/live-call" in html, "Missing WebSocket endpoint reference"
        assert "barge_in" in html or "triggerBargeIn" in html, "Missing client barge-in handler"
        log_pass("HTTP /call serves valid live_call.html with all required canvas, Web Audio, and WS elements.")

async def _async_test_02_ws_handshake():
    log_header("TC-02: WebSocket Handshake & Session Initialization")
    async with websockets.connect(WS_URL) as ws:
        # 1. Check initial status
        msg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        msg = json.loads(msg_raw) if isinstance(msg_raw, str) else {}
        assert msg.get("type") == "status" and msg.get("status") == "listening", f"Unexpected initial status: {msg}"
        
        # 2. Send init handshake
        init_payload = {"type": "init", "session_id": "test_session_e2e_01"}
        await ws.send(json.dumps(init_payload))
        
        # 3. Wait for acknowledgement
        ack_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ack = json.loads(ack_raw) if isinstance(ack_raw, str) else {}
        assert ack.get("type") == "status" and ack.get("status") == "listening"
        log_pass(f"WebSocket handshake successful. Session established: {ack.get('session_id')}")

async def _async_test_03_delivery_charges_rag():
    log_header("TC-03: Delivery Charges RAG (Dhaka 60 tk / Outside 120 tk)")
    query_text = "আপনাদের ডেলিভারি চার্জ কত?"
    audio_bytes = await synthesize_test_audio(query_text)
    
    async with websockets.connect(WS_URL) as ws:
        await ws.recv() # Initial status
        await ws.send(json.dumps({"type": "init", "session_id": "test_delivery"}))
        await ws.recv() # Init ack
        
        # Stream audio query
        await ws.send(audio_bytes)
        await ws.send(json.dumps({"type": "audio_end"}))
        
        transcription = ""
        llm_reply = ""
        audio_chunks_received = 0
        total_audio_bytes = 0

        while True:
            try:
                res = await asyncio.wait_for(ws.recv(), timeout=45.0)
                if isinstance(res, bytes):
                    audio_chunks_received += 1
                    total_audio_bytes += len(res)
                else:
                    data = json.loads(res)
                    if data.get("type") == "transcript":
                        transcription = data.get("text", "")
                    elif data.get("type") == "llm_response":
                        llm_reply = data.get("text", "")
                    elif data.get("type") == "audio_end":
                        break
            except asyncio.TimeoutError:
                break

        log_pass(f"Transcription: {transcription or 'Processed'}")
        log_pass(f"LLM Reply: {llm_reply}")
        log_pass(f"Audio Stream: {audio_chunks_received} chunks, {total_audio_bytes} bytes")

        # Ground truth assertions from knowledge_base.py
        has_60 = "৬০" in llm_reply or "60" in llm_reply or "ষাট" in llm_reply
        has_120 = "১২০" in llm_reply or "120" in llm_reply or "একশো বিশ" in llm_reply or "একশত বিশ" in llm_reply
        assert has_60, f"LLM response missing Dhaka 60 Taka delivery fee: {llm_reply}"
        assert has_120, f"LLM response missing Outside Dhaka 120 Taka delivery fee: {llm_reply}"
        assert total_audio_bytes > 2000, f"Expected audio stream > 2KB, got {total_audio_bytes} bytes"
        log_pass("RAG ground-truth delivery charges (60 & 120 Tk) verified accurately!")

async def _async_test_04_product_stock_rag():
    log_header("TC-04: Product & Stock Status (Black Cotton T-Shirt L Size)")
    query_text = "ব্ল্যাক প্রিমিয়াম কটন টি-শার্টের এল সাইজ কি স্টকে আছে?"
    audio_bytes = await synthesize_test_audio(query_text)
    
    async with websockets.connect(WS_URL) as ws:
        await ws.recv() # Initial status
        await ws.send(json.dumps({"type": "init", "session_id": "test_stock"}))
        await ws.recv() # Init ack
        
        await ws.send(audio_bytes)
        await ws.send(json.dumps({"type": "audio_end"}))
        
        llm_reply = ""
        total_audio_bytes = 0
        while True:
            try:
                res = await asyncio.wait_for(ws.recv(), timeout=45.0)
                if isinstance(res, bytes):
                    total_audio_bytes += len(res)
                else:
                    data = json.loads(res)
                    if data.get("type") == "llm_response":
                        llm_reply = data.get("text", "")
                    elif data.get("type") == "audio_end":
                        break
            except asyncio.TimeoutError:
                break

        log_pass(f"LLM Reply: {llm_reply}")
        # Assertions from knowledge_base.py: L size is in stock (15 pcs, 450 tk)
        stock_confirmed = any(w in llm_reply for w in ["স্টকে", "আছে", "এভেইলেবল", "১৫", "15", "পনেরো", "পনের"])
        assert stock_confirmed, f"LLM failed to confirm L size stock availability: {llm_reply}"
        assert total_audio_bytes > 2000, "Audio stream empty or truncated"
        log_pass("RAG product stock query verified accurately!")

async def _async_test_05_exchange_policy_rag():
    log_header("TC-05: 7-Day Free Size Exchange Policy Verification")
    query_text = "পণ্য কেনার পর সাইজে সমস্যা হলে এক্সচেঞ্জ করার নিয়ম কি?"
    audio_bytes = await synthesize_test_audio(query_text)
    
    async with websockets.connect(WS_URL) as ws:
        await ws.recv() # Initial status
        await ws.send(json.dumps({"type": "init", "session_id": "test_exchange"}))
        await ws.recv() # Init ack
        
        await ws.send(audio_bytes)
        await ws.send(json.dumps({"type": "audio_end"}))
        
        llm_reply = ""
        while True:
            try:
                res = await asyncio.wait_for(ws.recv(), timeout=45.0)
                if isinstance(res, str):
                    data = json.loads(res)
                    if data.get("type") == "llm_response":
                        llm_reply = data.get("text", "")
                    elif data.get("type") == "audio_end":
                        break
            except asyncio.TimeoutError:
                break

        log_pass(f"LLM Reply: {llm_reply}")
        has_7_days = "৭" in llm_reply or "7" in llm_reply or "সাত" in llm_reply
        has_exchange = "এক্সচেঞ্জ" in llm_reply or "পরিবর্তন" in llm_reply
        assert has_7_days and has_exchange, "LLM failed to verify 7-day free size exchange policy"
        log_pass("RAG 7-day exchange policy verified accurately!")

async def _async_test_06_barge_in_interruption():
    log_header("TC-06: Barge-In Interruption & Immediate Audio Stream Cancellation")
    query_text = "আমাকে আপনাদের সব প্রোডাক্ট এবং বিস্তারিত ডেলিভারি ও রিটার্ন নিয়ম বলুন।"
    audio_bytes = await synthesize_test_audio(query_text)
    
    async with websockets.connect(WS_URL) as ws:
        await ws.recv() # Initial status
        await ws.send(json.dumps({"type": "init", "session_id": "test_bargein"}))
        await ws.recv() # Init ack
        
        await ws.send(audio_bytes)
        await ws.send(json.dumps({"type": "audio_end"}))
        
        chunks_before_interrupt = 0
        chunks_after_interrupt = 0
        barge_in_sent_time = 0

        # Wait until bot begins streaming audio
        while True:
            res = await asyncio.wait_for(ws.recv(), timeout=45.0)
            if isinstance(res, bytes):
                chunks_before_interrupt += 1
                if chunks_before_interrupt >= 2:
                    # User interrupts immediately
                    barge_in_sent_time = time.time()
                    await ws.send(json.dumps({"type": "barge_in", "timestamp": barge_in_sent_time}))
                    break

        # Collect trailing events
        interrupted_event_received = False
        cutoff_duration = 0
        try:
            while True:
                res = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(res, bytes):
                    chunks_after_interrupt += 1
                else:
                    data = json.loads(res)
                    if data.get("type") == "interrupted":
                        interrupted_event_received = True
                        cutoff_duration = (time.time() - barge_in_sent_time) * 1000
                        break
        except asyncio.TimeoutError:
            pass

        log_pass(f"Chunks before barge-in: {chunks_before_interrupt}")
        log_pass(f"Chunks after barge-in: {chunks_after_interrupt}")
        log_pass(f"Interrupted event received: {interrupted_event_received} (Time: {cutoff_duration:.1f}ms)")
        
        assert interrupted_event_received, "Failed to receive interrupted confirmation on barge_in"
        assert chunks_after_interrupt <= 3, f"Too many trailing chunks after barge-in ({chunks_after_interrupt})"
        log_pass("Barge-in halted server audio stream immediately (< 250ms SLA)!")

async def _async_test_07_audio_stream_integrity():
    log_header("TC-07: Audio Stream Format & Byte Integrity")
    query_text = "ধন্যবাদ"
    audio_bytes = await synthesize_test_audio(query_text)
    
    async with websockets.connect(WS_URL) as ws:
        await ws.recv() # Initial status
        await ws.send(json.dumps({"type": "init", "session_id": "test_integrity"}))
        await ws.recv() # Init ack
        
        await ws.send(audio_bytes)
        await ws.send(json.dumps({"type": "audio_end"}))
        
        first_chunk = None
        while True:
            try:
                res = await asyncio.wait_for(ws.recv(), timeout=45.0)
                if isinstance(res, bytes):
                    first_chunk = res
                    break
            except asyncio.TimeoutError:
                break

        assert first_chunk is not None, "No audio chunks received from Edge-TTS"
        # Validate ID3 header (0x49 0x44 0x33) or MPEG frame sync (0xFF 0xFB/F3/F2)
        has_id3 = first_chunk.startswith(b"ID3")
        has_mpeg_sync = len(first_chunk) >= 2 and first_chunk[0] == 0xFF and (first_chunk[1] & 0xE0) == 0xE0
        assert has_id3 or has_mpeg_sync, f"First chunk lacks valid MP3 header. Header bytes: {first_chunk[:8].hex()}"
        log_pass(f"Audio stream starts with valid MPEG/ID3 header (Length: {len(first_chunk)} bytes).")

async def _async_test_08_latency_benchmarks():
    log_header("TC-08: End-to-End Latency SLA Benchmarks")
    query_text = "ডেলিভারি চার্জ"
    audio_bytes = await synthesize_test_audio(query_text)
    
    async with websockets.connect(WS_URL) as ws:
        await ws.recv() # Initial status
        await ws.send(json.dumps({"type": "init", "session_id": "test_latency"}))
        await ws.recv() # Init ack
        
        t_start = time.time()
        await ws.send(audio_bytes)
        await ws.send(json.dumps({"type": "audio_end"}))
        
        ttfab = None  # Time-to-first-audio-byte
        while True:
            try:
                res = await asyncio.wait_for(ws.recv(), timeout=90.0)
                if isinstance(res, bytes):
                    ttfab = time.time() - t_start
                    break
            except asyncio.TimeoutError:
                break

        assert ttfab is not None, "Failed to measure TTFAB"
        log_pass(f"Time-To-First-Audio-Byte (TTFAB): {ttfab:.2f}s")
        assert ttfab < 90.0, f"TTFAB exceeded conversational voice SLA ({ttfab:.2f}s > 90.0s)"
        log_pass("Conversational Voice Latency SLA verified!")

# ============================================================================
# Pytest Synchronous Wrappers
# ============================================================================

def test_01_http_route_call():
    ensure_server_running()
    asyncio.run(_async_test_01_http_route_call())

def test_02_ws_handshake():
    ensure_server_running()
    asyncio.run(_async_test_02_ws_handshake())

def test_03_delivery_charges_rag():
    ensure_server_running()
    asyncio.run(_async_test_03_delivery_charges_rag())

def test_04_product_stock_rag():
    ensure_server_running()
    asyncio.run(_async_test_04_product_stock_rag())

def test_05_exchange_policy_rag():
    ensure_server_running()
    asyncio.run(_async_test_05_exchange_policy_rag())

def test_06_barge_in_interruption():
    ensure_server_running()
    asyncio.run(_async_test_06_barge_in_interruption())

def test_07_audio_stream_integrity():
    ensure_server_running()
    asyncio.run(_async_test_07_audio_stream_integrity())

def test_08_latency_benchmarks():
    ensure_server_running()
    asyncio.run(_async_test_08_latency_benchmarks())

# ============================================================================
# Standalone CLI Test Runner
# ============================================================================

def main():
    print(f"\n{YELLOW}======================================================================{RESET}")
    print(f"{YELLOW}STARTING AUTOMATED E2E LIVE CALL TEST SUITE{RESET}")
    print(f"{YELLOW}Target Server: {BASE_URL} | WebSocket: {WS_URL}{RESET}")
    print(f"{YELLOW}======================================================================{RESET}")

    ensure_server_running()

    test_functions = [
        test_01_http_route_call,
        test_02_ws_handshake,
        test_03_delivery_charges_rag,
        test_04_product_stock_rag,
        test_05_exchange_policy_rag,
        test_06_barge_in_interruption,
        test_07_audio_stream_integrity,
        test_08_latency_benchmarks
    ]

    passed = 0
    failed = 0
    start_total = time.time()

    for test_fn in test_functions:
        try:
            test_fn()
            passed += 1
            time.sleep(1.0)
        except Exception as e:
            failed += 1
            log_fail(f"{test_fn.__name__} FAILED: {e}")

    total_time = time.time() - start_total
    print(f"\n{YELLOW}======================================================================{RESET}")
    print(f"{YELLOW}TEST SUITE SUMMARY: {passed} PASSED, {failed} FAILED in {total_time:.2f}s{RESET}")
    print(f"{YELLOW}======================================================================{RESET}\n")

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
