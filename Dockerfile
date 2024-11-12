FROM croncorp/python-ffmpeg:3.11.4-slim-bullseye

WORKDIR /app

# Pip install
ADD src ./src
ADD pyproject.toml .
ADD setup.py .

RUN pip install . --no-cache-dir
