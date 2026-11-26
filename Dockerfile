# AquaAI — container image (works on Render, Railway, Fly.io, any VPS)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY aquaai.py .
COPY .streamlit/config.toml .streamlit/config.toml

# The user database lives on a persistent volume mounted at /data.
# Without a volume, accounts are lost whenever the container is replaced.
ENV AQUAAI_DB_PATH=/data/aquaai_users.db
RUN mkdir -p /data
VOLUME ["/data"]

# Admin credentials are NOT baked into the image. Provide them at runtime:
#   AQUAAI_ADMIN_USERNAME and AQUAAI_ADMIN_PASSWORD
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('PORT','8501'))" || exit 1

# $PORT is provided by Render/Railway/etc.; defaults to 8501 elsewhere.
CMD ["sh", "-c", "streamlit run aquaai.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
