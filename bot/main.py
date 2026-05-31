import os
import re
import json
import uuid
import asyncio
import httpx
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
import aio_pika
import logging
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp


class FilterHealthMetrics:
    def filter(self, record) -> bool:
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
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:32b-instruct-q2_K")

SYSTEM_PROMPT = """You are a helpful personal assistant.
You must never discuss, reveal, or speculate about:
- Server infrastructure, hostnames, IP addresses, or network topology
- Kubernetes, Docker, or any deployment details
- Linux commands that could cause damage
- Credentials, tokens, or secrets of any kind

If asked about any of the above, politely decline."""

YOUTUBE_PROMPT_TEMPLATE = """Here is a transcript from a YouTube video.

Title: {title}
URL: {url}
Language: {lang}

Transcript:
{transcript}

---

Please write a structured one-page essay from this transcript.
Respond in the same language as the transcript ({lang}).

Structure:
- A clear title
- A concise introduction
- Key points organized in sections
- A short conclusion
- The original video link at the end: {url}

Keep it informative and well-structured."""

YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
)

rabbitmq_connection: aio_pika.RobustConnection = None
rabbitmq_channel: aio_pika.Channel = None


def sanitize(text: str) -> str:
    if TELEGRAM_TOKEN:
        return text.replace(TELEGRAM_TOKEN, "***")
    return text


def log(event: str, **kwargs):
    sanitized = {k: sanitize(str(v)) for k, v in kwargs.items()}
    print(json.dumps({"event": event, **sanitized}, ensure_ascii=False), flush=True)


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return video_id
    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_video_info(url: str) -> tuple[str, str]:
    """Returns (title, language_code)."""
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "no_warnings": True,
            "extract_flat": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Unknown Title")
            lang = info.get("language") or info.get("default_audio_language") or "en"
            return title, lang
    except Exception as e:
        log("yt_dlp_error", url=url, error=str(e))
        return "Unknown Title", "en"


def fetch_transcript(video_id: str, lang: str) -> str:
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang, "en"])
    except NoTranscriptFound:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_generated_transcript([lang, "en"])
            entries = transcript.fetch()
        except Exception:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = next(iter(transcript_list))
            entries = transcript.fetch()
    return " ".join(e["text"] for e in entries)


async def build_youtube_prompt(url: str) -> tuple[str, str]:
    loop = asyncio.get_event_loop()

    video_id = extract_video_id(url)

    (title, lang), transcript = await asyncio.gather(
        loop.run_in_executor(None, fetch_video_info, url),
        loop.run_in_executor(None, fetch_transcript, video_id, "en"),
    )

    if lang != "en":
        transcript = await loop.run_in_executor(None, fetch_transcript, video_id, lang)

    prompt = YOUTUBE_PROMPT_TEMPLATE.format(
        title=title,
        url=url,
        transcript=transcript,
        lang=lang,
    )
    return prompt, title


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


async def publish_request(chat_id: int, prompt: str):
    request_id = str(uuid.uuid4())
    body = json.dumps({
        "prompt": prompt,
        "model": DEFAULT_MODEL,
        "request_id": request_id,
        "chat_id": chat_id,
    })

    await rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=body.encode(),
            correlation_id=request_id,
            reply_to=REPLY_QUEUE,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=REQUEST_QUEUE,
    )

    log("llm_request_published", request_id=request_id, chat_id=chat_id, prompt_len=len(prompt))


async def on_llm_response(message: aio_pika.IncomingMessage) -> None:
    async with message.process():
        try:
            body = json.loads(message.body)
            chat_id = body.get("chat_id")
            result = body.get("result")
            error = body.get("error")
            request_id = body.get("request_id")

            if not chat_id:
                log("response_missing_chat_id", request_id=request_id)
                return

            if error:
                log("llm_error_response", request_id=request_id, chat_id=chat_id, error=error)
                await send_message(chat_id, "❌ something went wrong, please try again")
            else:
                await send_message(chat_id, result)
                log("reply_sent", request_id=request_id, chat_id=chat_id,
                    model=body.get("model_used"), duration_s=body.get("duration_seconds"))

        except Exception as e:
            log("response_handler_error", error=str(e))


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

                yt_match = YOUTUBE_URL_PATTERN.search(text)
                if yt_match:
                    url = yt_match.group(0)
                    await send_message(chat_id, "🎬 fetching transcript...")
                    try:
                        prompt, title = await build_youtube_prompt(url)
                        log("youtube_transcript_fetched", chat_id=chat_id, url=url, title=title)
                        await send_message(chat_id, f"⏳ summarizing: {title}")
                        await publish_request(chat_id, prompt)
                    except TranscriptsDisabled:
                        await send_message(chat_id, "❌ transcripts are disabled for this video")
                    except ValueError as e:
                        await send_message(chat_id, f"❌ could not parse URL: {e}")
                    except Exception as e:
                        log("youtube_error", chat_id=chat_id, url=url, error=str(e))
                        await send_message(chat_id, "❌ failed to fetch transcript, please try again")
                    continue

                if len(text) > MAX_MESSAGE_LENGTH:
                    await send_message(chat_id, "⚠️ Message too long, please keep it under 2000 characters.")
                    continue

                await send_message(chat_id, "⏳ thinking...")
                await publish_request(chat_id, f"{SYSTEM_PROMPT}\n\nUser: {text}")

        except Exception as e:
            log("poll_error", error=str(e))
            await asyncio.sleep(5)


async def setup_consumer():
    global rabbitmq_channel
    rabbitmq_channel = await rabbitmq_connection.channel()
    await rabbitmq_channel.declare_queue(REQUEST_QUEUE, durable=True)
    reply_queue = await rabbitmq_channel.declare_queue(REPLY_QUEUE, durable=True)
    await reply_queue.consume(on_llm_response)
    log("consumer_registered", reply_queue=REPLY_QUEUE)


@app.get("/health")
def health():
    return {"healthy": True}


@app.on_event("startup")
async def startup():
    global rabbitmq_connection

    rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
    rabbitmq_connection.reconnect_callbacks.add(lambda *_: asyncio.create_task(setup_consumer()))

    await setup_consumer()

    log("startup", rabbitmq_url=RABBITMQ_URL, reply_queue=REPLY_QUEUE, model=DEFAULT_MODEL)
    asyncio.create_task(poll_loop())