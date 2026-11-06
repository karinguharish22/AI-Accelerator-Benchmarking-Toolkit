FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-tools.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY benchmarks/ /app/benchmarks/
ENTRYPOINT ["python", "-m", "benchmarks.load_test"]
