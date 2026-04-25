import os
import httpx
import asyncio
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://kbrain:11434")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

ollama_model: str | None = None


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


async def send_message(chat_id: int, text: str) -> int:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text})
        r.raise_for_status()
        return r.json()["result"]["message_id"]


async def edit_message(chat_id: int, message_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(f"{TELEGRAM_API}/editMessageText", json={
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        })


async def ask_ollama(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        return r.json()["response"]


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
                    message_id = await send_message(chat_id, "⏳ thinking...")
                    reply = await ask_ollama(text)
                    await edit_message(chat_id, message_id, reply)
        except Exception as e:
            print(f"poll error: {e}")
            await asyncio.sleep(5)


@app.get("/health")
def health():
    return {"healthy": True}


@app.on_event("startup")
async def startup():
    global ollama_model
    ollama_model = await get_model()
    print(f"using model: {ollama_model}")
    asyncio.create_task(poll_loop())