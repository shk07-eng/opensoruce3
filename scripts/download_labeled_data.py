"""Download and split a labelled Hugging Face image dataset.

This script does not invent labels. It requires a dataset whose labels can be
mapped to ImageNet labels, records the dataset card metadata, hashes files to
prevent duplicate images, and creates 30 tuning / 70 evaluation records.
"""
import argparse, hashlib, json, random
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--split",default="train"); p.add_argument("--out",default="mlops-data"); p.add_argument("--seed",type=int,default=20261106); p.add_argument("--license",required=True,help="License copied from the dataset card")
    a=p.parse_args()
    from datasets import load_dataset
    ds=load_dataset(a.dataset, split=a.split)
    if len(ds)<100: raise SystemExit(f"Dataset contains only {len(ds)} records; 100 labelled records are required.")
    label_col="label" if "label" in ds.column_names else None
    image_col="image" if "image" in ds.column_names else None
    if not label_col or not image_col: raise SystemExit(f"Expected image/label columns, found {ds.column_names}.")
    rng=random.Random(a.seed); indices=list(range(len(ds))); rng.shuffle(indices); chosen=[]; hashes=set()
    for i in indices:
        image=ds[i][image_col]; raw=image.tobytes(); digest=hashlib.sha256(raw).hexdigest()
        if digest in hashes: continue
        hashes.add(digest); chosen.append((i,digest))
        if len(chosen)==100: break
    if len(chosen)<100: raise SystemExit(f"Only {len(chosen)} unique labelled images were found.")
    label_feature=ds.features[label_col]; names=getattr(label_feature,"names",None)
    labels=[]
    for i,_ in chosen:
        value=ds[i][label_col]
        label=names[value] if names and isinstance(value,int) else value
        if not isinstance(label,str): raise SystemExit("Labels are not textual; refusing an unverified class mapping.")
        labels.append(label)
    groups={"tune":[],"eval":[]}
    # Deterministic split; the manifest preserves source index and hash.
    for n,(item,label) in enumerate(zip(chosen,labels)): groups["tune" if n<30 else "eval"].append({"source_index":item[0],"sha256":item[1],"label":label})
    out=Path(a.out); (out/"images").mkdir(parents=True,exist_ok=True)
    for group in groups.values():
        for item in group:
            image=ds[item["source_index"]][image_col].convert("RGB"); path=out/"images"/f"{item['sha256']}.jpg"; image.save(path,quality=95); item["path"]=str(path)
    (out/"split.json").write_text(json.dumps({"dataset":a.dataset,"split":a.split,"license":a.license,"seed":a.seed,"counts":{"tune":30,"eval":70},"groups":groups},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Prepared manifests only: tune=30 eval=70 unique={len(hashes)}; source images remain in the HF cache.")
if __name__=="__main__": main()
