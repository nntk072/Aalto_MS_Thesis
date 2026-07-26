FROM python:3.12-slim AS builder

COPY zscaler.crt /usr/local/share/ca-certificates/zscaler.crt
RUN update-ca-certificates \
    && sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.sources 2>/dev/null || true \
    && sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN pip install uv

WORKDIR /build

RUN python -m venv .venv

RUN curl -k -fsSL -o /tmp/torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl \
    "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" \
    && uv pip install --python .venv /tmp/torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl \
    && rm /tmp/torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

RUN curl -k -fsSL https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    | tar -xz -C /tmp \
    && cd /tmp/ta-lib-0.6.4 \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/ta-lib-0.6.4

COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime

COPY zscaler.crt /usr/local/share/ca-certificates/zscaler.crt
RUN update-ca-certificates \
    && sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.sources 2>/dev/null || true \
    && sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true

RUN groupadd -g 1001 appgroup && useradd -u 1001 -g appgroup -m -d /app appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder --chown=1001:1001 /usr/local/lib/libta_lib* /usr/local/lib/
RUN ldconfig /usr/local/lib

COPY --from=builder --chown=1001:1001 /build /app
COPY --from=builder --chown=1001:1001 /build/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

USER 1001

ENTRYPOINT ["python"]

FROM runtime AS dev

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
USER 1001

FROM runtime AS test

USER 1001
ENTRYPOINT ["python", "-m", "pytest"]
