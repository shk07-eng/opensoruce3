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
app = FastAPI(title="Animal Classifier", version="2.0.0")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()
preprocess = weights.transforms()
labels = weights.meta["categories"]

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "ResNet18-ImageNet",
        "model_version": "2.0",
        "device": "cpu"
    }

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "JPG, PNG, WEBP ???대?吏 ?뚯씪留??낅줈?쒗븷 ???덉뒿?덈떎.")
    data = await file.read()
    if not data: raise HTTPException(400, "鍮??뚯씪? 遺꾩꽍?????놁뒿?덈떎.")
    if len(data) > MAX_BYTES: raise HTTPException(413, "?뚯씪 ?ш린??10MB ?댄븯留??덉슜?⑸땲??")
    try: image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError: raise HTTPException(400, "?좏슚???대?吏 ?뚯씪???쎌쓣 ???놁뒿?덈떎.")
    with torch.inference_mode():
        probabilities = torch.softmax(model(preprocess(image).unsqueeze(0)), dim=1)[0]
    cat_ids = [
        i for i, name in enumerate(labels)
        if any(word in name.lower() for word in (
            "tabby", "tiger cat", "persian cat",
            "siamese cat", "egyptian cat"
        ))
    ]

    dog_ids = [
        i for i, name in enumerate(labels)
        if any(word in name.lower() for word in (
            "dog", "terrier", "retriever", "spaniel",
            "poodle", "hound", "shepherd", "schnauzer",
            "pinscher", "chihuahua", "corgi", "collie",
            "malamute", "husky", "beagle", "boxer",
            "bulldog", "dalmatian", "mastiff", "pug",
            "rottweiler", "samoyed", "wolfhound"
        ))
    ]

    cat_score = float(probabilities[cat_ids].sum())
    dog_score = float(probabilities[dog_ids].sum())

    species = "cat" if cat_score >= dog_score else "dog"
    confidence = max(cat_score, dog_score)

    return {
        "filename": file.filename,
        "predictions": [
            {
                "label": species,
                "confidence": round(confidence * 100, 2)
            }
        ],
        "model_version": "2.0",
        "estimate": True,
        "notice": "AI 분석 결과는 참고용 추정치입니다."
    }

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
@app.get("/")
def index(): return FileResponse(ROOT / "static" / "index.html")


