# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY Makefile/ Makefile/
COPY pyproject.toml pyproject.toml
COPY toxic_comments/ toxic_comments/
#COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
#Add commands to work with gcp

#ENTRYPOINT ["python", "-u", "toxic_comments/train_model.py"]
ENTRYPOINT [ "make" "data" ]