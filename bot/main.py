import os
import json
import time
import uuid
import logging
import asyncio
import httpx
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
import aio_pika

class FilterHealthMetrics(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/health" not in msg and "/metrics" not in msg

logging.getLogger("uvicorn.access").addFilter(FilterHealthMetrics())

app = FastAPI()
Instrumentator().instrument(app).expose(app)

llm_request_duration = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration via llm-broker",
    ["model"]
)

unauthorized_attempts = Counter(
    "unauthorized_attempts_total",
    "Number of unauthorized access attempts"
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq.rabbitmq.svc.cluster.local/")
REQUEST_QUEUE = "llm_requests"
REPLY_QUEUE = "llm_responses_telegram_bot"
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
ALLOWED_USER_IDS = {int(uid) for uid in os.getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip()}
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0")) or None
MAX_MESSAGE_LENGTH = 2000
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:32b-instruct-q2_K")

SYSTEM_PROMPT = """You are a helpful personal assistant.
You must never discuss, reveal, or speculate about:
- Server infrastructure, hostnames, IP addresses, or network topology
- Kubernetes, Docker, or any deployment details
- Linux commands that could cause damage
- Credentials, tokens, or secrets of any kind

If asked about any of the above, politely decline."""

# correlation_id → asyncio.Future
pending: dict[str, asyncio.Future] = {}

rabbitmq_connection: aio_pika.RobustConnection = None
rabbitmq_channel: aio_pika.Channel = None


def sanitize(text: str) -> str:
    if TELEGRAM_TOKEN:
        return text.replace(TELEGRAM_TOKEN, "***")
    return text


def log(event: str, **kwargs):
    sanitized = {k: sanitize(str(v)) for k, v in kwargs.items()}
    print(json.dumps({"event": event, **sanitized}, ensure_ascii=False), flush=True)


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


async def ask_llm_broker(prompt: str) -> tuple[str, float, str]:
    """Publish to llm_requests, wait for response on llm_responses_telegram_bot.
    Returns (result, duration_seconds, model_used)."""
    correlation_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    pending[correlation_id] = future

    body = json.dumps({
        "prompt": f"{SYSTEM_PROMPT}\n\nUser: {prompt}",
        "model": DEFAULT_MODEL,
        "request_id": correlation_id,
    })

    t0 = time.monotonic()
    log("llm_request_published", correlation_id=correlation_id, prompt_len=len(prompt))

    await rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=body.encode(),
            correlation_id=correlation_id,
            reply_to=REPLY_QUEUE,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=REQUEST_QUEUE,
    )

    try:
        response = await asyncio.wait_for(future, timeout=LLM_TIMEOUT)
    finally:
        pending.pop(correlation_id, None)

    duration = time.monotonic() - t0
    return response["result"], duration, response.get("model_used", DEFAULT_MODEL)


async def on_llm_response(message: aio_pika.IncomingMessage) -> None:
    async with message.process():
        try:
            body = json.loads(message.body)
            correlation_id = message.correlation_id
            future = pending.get(correlation_id)
            if future and not future.done():
                if body.get("error"):
                    future.set_exception(Exception(body["error"]))
                else:
                    future.set_result(body)
                log("llm_response_received", correlation_id=correlation_id,
                    duration_s=body.get("duration_seconds"), model=body.get("model_used"))
            else:
                log("llm_response_no_waiter", correlation_id=correlation_id)
        except Exception as e:
            log("llm_response_parse_error", error=str(e))


async def handle_message(chat_id: int, user: dict, text: str):
    log("message_received", chat_id=chat_id, user=user, text_len=len(text))
    try:
        result, duration, model = await ask_llm_broker(text)
        llm_request_duration.labels(model=model).observe(duration)
        log("reply_sent", chat_id=chat_id, duration_s=round(duration, 2), model=model)
        await send_message(chat_id, result)
    except asyncio.TimeoutError:
        log("llm_timeout", chat_id=chat_id)
        await send_message(chat_id, "⏰ timeout, please try again")
    except Exception as e:
        log("reply_error", chat_id=chat_id, error=str(e))
        await send_message(chat_id, "❌ something went wrong, please try again")


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
                    log("unauthorized_user", user_id=user_id, username=user.get("username"))
                    unauthorized_attempts.inc()
                    await notify_admin(user, text)
                    await send_message(chat_id, "Sorry, you are not authorized to use this bot.")
                    continue

                if len(text) > MAX_MESSAGE_LENGTH:
                    await send_message(chat_id, "⚠️ Message too long, please keep it under 2000 characters.")
                    continue

                queue_size = len(pending)
                if queue_size > 0:
                    await send_message(chat_id, f"⏳ thinking... ({queue_size + 1} requests in queue)")
                else:
                    await send_message(chat_id, "⏳ thinking...")

                asyncio.create_task(handle_message(chat_id, user, text))
                log("message_queued", chat_id=chat_id, pending=queue_size + 1)

        except Exception as e:
            log("poll_error", error=str(e))
            await asyncio.sleep(5)


@app.get("/health")
def health():
    return {"healthy": True}


@app.on_event("startup")
async def startup():
    global rabbitmq_connection, rabbitmq_channel

    rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
    rabbitmq_channel = await rabbitmq_connection.channel()

    # declare both queues so they exist regardless of start order
    await rabbitmq_channel.declare_queue(REQUEST_QUEUE, durable=True)
    reply_queue = await rabbitmq_channel.declare_queue(REPLY_QUEUE, durable=True)

    await reply_queue.consume(on_llm_response)

    log("startup", rabbitmq_url=RABBITMQ_URL, reply_queue=REPLY_QUEUE, model=DEFAULT_MODEL)
    asyncio.create_task(poll_loop())