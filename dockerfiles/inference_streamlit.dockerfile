# Base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_inference.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY toxic_comments/ toxic_comments/
COPY models/production/production_quantized.ckpt models/production/production_quantized.ckpt

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PYTHONPATH "${PYTHONPATH}:/toxic_comments/models"

EXPOSE 8501/tcp

ENTRYPOINT ["streamlit", "run", "./toxic_comments/api/streamlit_input_inference.py"]