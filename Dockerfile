FROM python:3.12-slim-bookworm
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY --chown=10001:10001 src/ src/
RUN --mount=type=secret,id=build_ca,target=/tmp/build-ca.pem,required=false \
    if [ -f /tmp/build-ca.pem ]; then export PIP_CERT=/tmp/build-ca.pem SSL_CERT_FILE=/tmp/build-ca.pem; fi; \
    python -m pip install --no-cache-dir uv==0.13.6 && uv sync --frozen --no-dev --native-tls
COPY --chown=10001:10001 data/ data/
RUN mkdir -p /models && chown 10001:10001 /models
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false
USER 10001:10001
ENTRYPOINT ["hybrid-search"]
CMD ["serve", "--host", "0.0.0.0"]
