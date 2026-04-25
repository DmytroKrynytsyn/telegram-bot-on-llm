import os
import json
import time
import logging
import httpx
import asyncio
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

class FilterHealthMetrics(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/health" not in msg and "/metrics" not in msg

logging.getLogger("uvicorn.access").addFilter(FilterHealthMetrics())

app = FastAPI()
Instrumentator().instrument(app).expose(app)

llm_request_duration = Histogram(
    "llm_request_duration_seconds",
    "Ollama LLM request duration",
    ["model"]
)

unauthorized_attempts = Counter(
    "unauthorized_attempts_total",
    "Number of unauthorized access attempts"
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://kbrain:11434")
OLLAMA_TIMEOUT = 30 * 60
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
ALLOWED_USER_IDS = {int(uid) for uid in os.getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip()}
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0")) or None
MAX_MESSAGE_LENGTH = 2000

SYSTEM_PROMPT = """You are a helpful personal assistant.
You must never discuss, reveal, or speculate about:
- Server infrastructure, hostnames, IP addresses, or network topology
- Kubernetes, Docker, or any deployment details
- Linux commands that could cause damage
- Credentials, tokens, or secrets of any kind

If asked about any of the above, politely decline."""

ollama_model: str | None = None
ollama_queue: asyncio.Queue = None


def sanitize(text: str) -> str:
    if TELEGRAM_TOKEN:
        return text.replace(TELEGRAM_TOKEN, "***")
    return text


def log(event: str, **kwargs):
    sanitized = {k: sanitize(str(v)) for k, v in kwargs.items()}
    print(json.dumps({"event": event, **sanitized}, ensure_ascii=False), flush=True)


async def get_model() -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        r.raise_for_status()
        return r.json()["models"][0]["name"]


async def get_updates(offset: int | None = None):
    params = {"timeout": 30}
    if offset is not None:
        params["offset"] = offset
    async with httpx.AsyncClient(timeout=40) as client:
        r = await client.get(f"{TELEGRAM_API}/getUpdates", params=params)
        r.raise_for_status()
        return r.json().get("result", [])


async def send_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text})


async def notify_admin(user: dict, text: str):
    if not ADMIN_CHAT_ID:
        return
    msg = (
        f"🚫 Unauthorized access attempt\n\n"
        f"User info:\n<pre>{json.dumps(user, ensure_ascii=False, indent=2)}</pre>\n\n"
        f"Message: {text[:500]}"
    )
    async with httpx.AsyncClient() as client:
        await client.post(f"{TELEGRAM_API}/sendMessage", json={
            "chat_id": ADMIN_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        })


async def ask_ollama(prompt: str) -> str:
    log("llm_request_started", model=ollama_model, prompt_len=len(prompt))
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": ollama_model,
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            r.raise_for_status()
            reply = r.json()["response"]
            duration = round(time.monotonic() - t0, 2)
            llm_request_duration.labels(model=ollama_model).observe(duration)
            log("llm_response_received", model=ollama_model, duration_s=duration, reply_len=len(reply))
            return reply
    except httpx.TimeoutException:
        duration = round(time.monotonic() - t0, 2)
        log("llm_timeout", model=ollama_model, duration_s=duration)
        raise
    except Exception as e:
        duration = round(time.monotonic() - t0, 2)
        log("llm_request_failed", model=ollama_model, duration_s=duration, error=str(e))
        raise


async def handle_message(chat_id: int, user: dict, text: str):
    log("message_received", chat_id=chat_id, user=user, text_len=len(text), text=text)
    try:
        reply = await ask_ollama(text)
        await send_message(chat_id, reply)
        log("reply_sent", chat_id=chat_id, user=user, reply_len=len(reply), reply=reply)
    except httpx.TimeoutException:
        await send_message(chat_id, "⏰ timeout, please try again")
    except Exception as e:
        log("reply_error", chat_id=chat_id, error=str(e))
        await send_message(chat_id, "❌ something went wrong, please try again")


async def ollama_worker():
    while True:
        chat_id, user, text = await ollama_queue.get()
        try:
            await handle_message(chat_id, user, text)
        finally:
            ollama_queue.task_done()


async def poll_loop():
    offset = None
    while True:
        try:
            updates = await get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message", {})
                chat_id = message.get("chat", {}).get("id")
                user = message.get("from", {})
                user_id = user.get("id")
                text = message.get("text", "").strip()

                if not chat_id or not text:
                    continue

                if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
                    log("unauthorized_user", user_id=user_id, username=user.get("username"), user=user, text=text)
                    unauthorized_attempts.inc()
                    await notify_admin(user, text)
                    await send_message(chat_id, "Sorry, you are not authorized to use this bot.")
                    continue

                if len(text) > MAX_MESSAGE_LENGTH:
                    await send_message(chat_id, "⚠️ Message too long, please keep it under 2000 characters.")
                    continue

                queue_size = ollama_queue.qsize()
                if queue_size > 0:
                    await send_message(chat_id, f"⏳ thinking... ({queue_size + 1} requests in queue)")
                else:
                    await send_message(chat_id, "⏳ thinking...")

                await ollama_queue.put((chat_id, user, text))
                log("message_queued", chat_id=chat_id, user=user, queue_size=queue_size + 1)

        except Exception as e:
            log("poll_error", error=str(e))
            await asyncio.sleep(5)


@app.get("/health")
def health():
    return {"healthy": True}


@app.on_event("startup")
async def startup():
    global ollama_model, ollama_queue
    ollama_queue = asyncio.Queue()
    ollama_model = await get_model()
    log("startup", model=ollama_model, ollama_url=OLLAMA_URL, allowed_users=list(ALLOWED_USER_IDS))
    asyncio.create_task(ollama_worker())
    asyncio.create_task(poll_loop())