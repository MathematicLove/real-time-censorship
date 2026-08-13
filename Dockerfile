FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py ./

RUN useradd --create-home --uid 10001 censor \
    && mkdir -p /srv/logs /srv/data \
    && chown -R censor:censor /srv
USER censor

ENV NSFW_API_HOST=0.0.0.0 \
    NSFW_API_PORT=8000 \
    NSFW_LOG_DIR=/srv/logs \
    NSFW_DATA_DIR=/srv/data

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "main.py"]