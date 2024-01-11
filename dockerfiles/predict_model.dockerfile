# Base image
FROM python:3.11-slim

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY toxic_comments/ toxic_comments/
COPY data/ data/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

#ENTRYPOINT ["python", "-u", "toxic_comments/predict_model.py"]