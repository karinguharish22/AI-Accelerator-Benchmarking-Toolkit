FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    PJRT_DEVICE=TPU

# CPU build of torch + the matching torch_xla wheel with libtpu.
RUN pip install --no-cache-dir torch==2.4.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "torch_xla[tpu]==2.4.0" -f https://storage.googleapis.com/libtpu-releases/index.html

WORKDIR /app
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY training/ /app/training/
ENTRYPOINT ["python", "-m", "training.train"]
