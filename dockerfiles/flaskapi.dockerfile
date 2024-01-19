FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y nano

COPY requirements_inference.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY toxic_comments/ toxic_comments/
COPY models/production/model.ckpt models/production/model.ckpt

RUN pip install -r requirements.txt 
RUN pip install . --no-deps 

ENV PYTHONPATH "${PYTHONPATH}:/toxic_comments/models"

EXPOSE 5000

ENTRYPOINT ["python", "./toxic_comments/fapi/app.py"]

#docker run -it -p 5000:5000 fla:latest /bin/bash