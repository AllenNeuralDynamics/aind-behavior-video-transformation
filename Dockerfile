FROM lscr.io/linuxserver/ffmpeg:8.0.1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Pip install
ADD src ./src
ADD pyproject.toml .
ADD setup.py .

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install . --no-cache-dir

ENTRYPOINT []
