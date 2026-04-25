FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /build

COPY bot/pyproject.toml .
COPY bot/main.py ./bot/

RUN uv build --wheel --out-dir /dist


FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /dist/*.whl .

RUN pip install --no-cache-dir *.whl && rm *.whl

EXPOSE 8000

CMD ["uvicorn", "bot.main:app", "--host", "0.0.0.0", "--port", "8000"]