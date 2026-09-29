# One image for the detector API (C14) and the dashboard (C16).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libpcap0.8 tcpreplay iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image small; the pinned version matches requirements.txt.
RUN pip install --no-cache-dir torch==$(grep -oP '^torch==\K.*' requirements.txt) \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY dashboard/ dashboard/
RUN pip install --no-cache-dir --no-deps -e .

ENV DG_DATA=/app/data DG_MODELS=/app/models
EXPOSE 8000 8501
CMD ["uvicorn", "xnids.live.api:app", "--host", "0.0.0.0", "--port", "8000"]
