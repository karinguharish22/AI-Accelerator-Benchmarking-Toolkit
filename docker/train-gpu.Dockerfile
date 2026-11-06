FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY training/ /app/training/
ENTRYPOINT ["python", "-m", "training.train"]
