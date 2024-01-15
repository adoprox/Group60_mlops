FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY slowapi/requirements.txt /

RUN pip install -r requirements.txt --no-cache-dir

COPY toxic_comments/ toxic_comments/
COPY models/bert-toxic-classifier/models_bert-toxic-classifier_epoch=1-val_loss=0.15.ckpt /models/bertmodels/bert-toxic-classifier/
COPY slowapi/ slowapi/

EXPOSE 80

CMD ["uvicorn", "slowapi.main:app", "--host", "0.0.0.0", "--port", "80"]

#docker run -p 8000:80 hosting uvicorn slowapi.main:app --host 0.0.0.0 --port 80