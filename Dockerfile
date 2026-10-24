FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/code
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user app ./app
USER user
WORKDIR /home/user/code/app
# Hugging Face Spaces expects port 7860; Render/Cloud Run pass $PORT.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
