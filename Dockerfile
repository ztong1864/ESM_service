FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Exact versions of the tested environment. torch comes from PyTorch's CPU index: the Linux build on PyPI
# pulls in several GB of CUDA libraries that a CPU server never uses. libgomp1 (OpenMP runtime) is what
# torch needs at run time; build-essential is only needed while installing and goes in the same layer.
COPY requirements.lock.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# The model files are mounted, not baked in: they are hundreds of MB and are fetched once with
# scripts/fetch_models.py (see docker-compose.yml).
ENV ESM_MODELS_DIR=/app/models

EXPOSE 8789

CMD ["python", "scripts/run_esm_api.py", "--host", "0.0.0.0", "--port", "8789"]
