# MLOps evaluation notes

The current production model is `torchvision` ResNet-18 with ImageNet pretrained weights. Torchvision supplies the supported 1,000 ImageNet labels and the default 224px resize/center-crop plus ImageNet normalization. The existing API returns top-3 estimates; it does not expose ground-truth accuracy.

The scripts in `scripts/` deliberately stop unless a real labelled image dataset, a dataset-card license, unique hashes, and local image paths are available. They never synthesize labels. Run them only after reviewing the dataset card:

```powershell
python -m pip install -r requirements-mlops.txt
python scripts/download_labeled_data.py DATASET_ID --license "LICENSE FROM DATASET CARD"
python scripts/evaluate_model.py mlops-data/split.json --group tune --model-version 1.0
```

The evaluation set is never passed to the tuning command. Do not commit `mlops-data/`, model caches, images, or result logs; they are excluded by `.gitignore`.
