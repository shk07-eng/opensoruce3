"""Evaluate the existing ResNet-18 on a manifest; no tuning data is mixed with eval."""
import argparse, json, time
from pathlib import Path
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

def main():
    p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("--group",choices=["tune","eval"],required=True); p.add_argument("--model-version",default="1.0"); p.add_argument("--output",default="mlops-results"); p.add_argument("--strategy",choices=["top1","aggregate"],default="top1"); a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text(encoding="utf-8")); rows=m["groups"][a.group]
    if len(rows)!=(30 if a.group=="tune" else 70): raise SystemExit("Manifest size is not the required 30/70 split.")
    weights=ResNet18_Weights.DEFAULT; model=resnet18(weights=weights).eval(); preprocess=weights.transforms(); labels=weights.meta["categories"]; predictions=[]
    for row in rows:
        # The manifest must be extended with a local image path by the data preparation step.
        if "path" not in row: raise SystemExit("No local image path in manifest; refusing to evaluate unknown data.")
        image=Image.open(row["path"]).convert("RGB"); start=time.perf_counter()
        with torch.inference_mode(): scores=torch.softmax(model(preprocess(image).unsqueeze(0)),1)[0]
        value,index=torch.max(scores,0); predicted=labels[int(index)]
        confidence=float(value)
        if a.strategy == "aggregate":
            cat_ids=[i for i,n in enumerate(labels) if any(x in n.lower() for x in ("tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat"))]
            dog_ids=[i for i,n in enumerate(labels) if any(x in n.lower() for x in ("dog", "terrier", "retriever", "spaniel", "poodle", "hound", "shepherd", "schnauzer", "pinscher", "chihuahua", "corgi", "collie", "malamute", "husky", "beagle", "boxer", "bulldog", "dalmatian", "dane", "mastiff", "pekinese", "pug", "rottweiler", "samoyed", "wolfhound"))]
            cat_score=float(scores[cat_ids].sum()); dog_score=float(scores[dog_ids].sum())
            predicted="cat" if cat_score >= dog_score else "dog"; confidence=max(cat_score,dog_score)
        predictions.append({**row,"predicted":predicted,"confidence":round(confidence,6),"strategy":a.strategy,"inference_ms":round((time.perf_counter()-start)*1000,3),"model_version":a.model_version})
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); (out/f"{a.group}-{a.model_version}.json").write_text(json.dumps({"group":a.group,"model_version":a.model_version,"strategy":a.strategy,"count":len(predictions),"predictions":predictions},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()
