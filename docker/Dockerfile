FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Omnia SDK
COPY python/ /app/python/
RUN pip install /app/python/

# Copy core libs
COPY cli/build/omnia-cli /usr/local/bin/omnia-cli 2>/dev/null || true

ENTRYPOINT ["omnia"]
CMD ["--help"]
