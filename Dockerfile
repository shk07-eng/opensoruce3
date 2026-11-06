FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/torch-cache \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app
COPY requirements.txt requirements-model.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements-model.txt
COPY app ./app
COPY static ./static
RUN mkdir -p "$TORCH_HOME" \
    && python -m compileall -q app \
    && python -c "from app.main import model, preprocess; import torch; x=torch.zeros(1,3,224,224); y=model(x); assert y.shape[-1] == 1000; print('ResNet18 CPU model loaded and inference verified')"
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host ${HOST} --port ${PORT}"]
