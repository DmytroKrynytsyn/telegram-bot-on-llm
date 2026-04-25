import os
import json
import time
import httpx
import asyncio
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

app = FastAPI()
Instrumentator().instrument(app).expose(app)

llm_request_duration = Histogram(
    "llm_request_duration_seconds",
    "Ollama LLM request duration",
    ["model"]
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://kbrain:11434")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
OLLAMA_TIMEOUT = 30 * 60

ollama_model: str | None = None


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


async def ask_ollama(prompt: str) -> str:
    log("llm_request_started", model=ollama_model, prompt_len=len(prompt))
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False},
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


async def poll_loop():
    offset = None
    while True:
        try:
            updates = await get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message", {})
                chat_id = message.get("chat", {}).get("id")
                text = message.get("text", "").strip()
                if chat_id and text:
                    log("message_received", chat_id=chat_id, text_len=len(text), text=text)
                    await send_message(chat_id, "⏳ thinking...")
                    try:
                        reply = await ask_ollama(text)
                        await send_message(chat_id, reply)
                        log("reply_sent", chat_id=chat_id, reply_len=len(reply), reply=reply)
                    except httpx.TimeoutException:
                        await send_message(chat_id, "⏰ timeout, please try again")
                    except Exception as e:
                        log("reply_error", chat_id=chat_id, error=str(e))
                        await send_message(chat_id, "❌ something went wrong, please try again")
        except Exception as e:
            log("poll_error", error=str(e))
            await asyncio.sleep(5)


@app.get("/health")
def health():
    return {"healthy": True}


@app.on_event("startup")
async def startup():
    global ollama_model
    ollama_model = await get_model()
    log("startup", model=ollama_model, ollama_url=OLLAMA_URL)
    asyncio.create_task(poll_loop())