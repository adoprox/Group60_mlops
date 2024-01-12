# Base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /

ENV WANDB_API_KEY=''

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY Makefile/ Makefile/
COPY pyproject.toml pyproject.toml
COPY toxic_comments/ toxic_comments/

RUN pip install -r requirements.txt --no-cache-dir
#Add commands to work with gcp

#ENTRYPOINT ["python", "-u", "toxic_comments/train_model.py"]