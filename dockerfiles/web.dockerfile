FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y nano

COPY toxic_comments/slowapi/requirements.txt /

RUN pip install -r requirements.txt --no-cache-dir

COPY toxic_comments/ toxic_comments/
COPY /toxic_comments/slowapi /slowapi
COPY models/bert-toxic-classifier/models_bert-toxic-classifier_epoch=1-val_loss=0.15.ckpt models/bert-toxic-classifier/

ENV PYTHONPATH=/

EXPOSE 5000

#ENTRYPOINT [ "python" ]
#CMD [ "slowapi/ask.py" ]

#docker run -p 5000:5000 name_of_your_image
#docker run -p 8000:80 hosting uvicorn slowapi.main:app --host 0.0.0.0 --port 80
#change run.app(host='0.0.0.0', port=5000)