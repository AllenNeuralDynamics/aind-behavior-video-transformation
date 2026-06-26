FROM docker.io/mwader/static-ffmpeg:8.1.2-amd64 AS ffmpeg
FROM docker.io/python:3.13-slim-trixie

COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
