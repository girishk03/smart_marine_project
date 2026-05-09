FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
  && pip install -r requirements.txt

COPY . .

EXPOSE 8000

ENV SMART_MARINE_MODEL_PATH=/app/models/ocean_waste_model_m2/weights/best.pt

CMD ["python", "api_server.py"]
