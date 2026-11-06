import io
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 10 * 1024 * 1024
app = FastAPI(title="동물 종류 판별기", version="1.0.0")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()
preprocess = weights.transforms()
labels = weights.meta["categories"]

@app.get("/api/health")
def health():
    return {"status": "ok", "model": "ResNet18-ImageNet", "device": "cpu"}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "JPG, PNG, WEBP 등 이미지 파일만 업로드할 수 있습니다.")
    data = await file.read()
    if not data: raise HTTPException(400, "빈 파일은 분석할 수 없습니다.")
    if len(data) > MAX_BYTES: raise HTTPException(413, "파일 크기는 10MB 이하만 허용됩니다.")
    try: image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError: raise HTTPException(400, "유효한 이미지 파일을 읽을 수 없습니다.")
    with torch.inference_mode():
        probabilities = torch.softmax(model(preprocess(image).unsqueeze(0)), dim=1)[0]
    values, indices = torch.topk(probabilities, 3)
    predictions = [{"label": labels[int(i)], "confidence": round(float(v) * 100, 2)} for v, i in zip(values, indices)]
    return {"filename": file.filename, "predictions": predictions, "estimate": True, "notice": "AI 분석 결과는 참고용 추정치이며 전문적인 판정이 아닙니다."}

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
@app.get("/")
def index(): return FileResponse(ROOT / "static" / "index.html")
