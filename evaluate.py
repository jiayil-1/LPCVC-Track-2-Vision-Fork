import argparse
import numpy as np
import os 
import h5py
import torch
import json

parser = argparse.ArgumentParser(description="Evaluate QAI Hub inference logits.")
parser.add_argument(
    "--h5",
    default="dataset-export-qnn16.h5",
    help="Path to H5 logits file downloaded from QAI Hub inference output.",
)
args = parser.parse_args()

filename = args.h5
data_path = "./preprocessed/test16"
manifest_path = os.path.join(data_path, "manifest.jsonl")
class_map_path = "class_map.json"

logits = []

with h5py.File(filename, "r") as f:
    grp = f["data/0"]

    sorted_keys = sorted(grp.keys(), key=lambda x: int(x.split("_")[1]))

    for k in sorted_keys:
        x = grp[k][...]          # shape (1, 92)
        logits.append(x.squeeze())  # -> (92,)
    if sorted_keys:
        print("First H5 output key:", sorted_keys[0])

logits = np.stack(logits, axis=0)  # shape (N, 92)

# print("Final logits shape:", logits.shape)
# print(logits.dtype)

logits = torch.as_tensor(logits, dtype=torch.float32)
print("Logits: ", logits[0:5])
# logits = logits.squeeze()
# print(logits[0:5])
print(logits.shape)
probs = torch.softmax(logits, dim=1)
print(probs[0])
# Recreate labels in the same order used by run_inference.py
labels = []
with open(class_map_path, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {i: cls for cls, i in class_to_idx.items()}
if not os.path.exists(manifest_path):
    raise FileNotFoundError(f"'{manifest_path}' not found. Run preprocess_and_save.py first.")

with open(manifest_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
    for record in records:
        labels.append(class_to_idx[record["label"]])

if records:
    print("First input tensor path:", records[0]["tensor_path"])

labels = torch.tensor(labels, dtype=torch.int64)


n_logits = probs.shape[0]
if labels.shape[0] > n_logits:
    print(f"ℹ️  H5 has {n_logits} results but manifest has {labels.shape[0]} labels — "
          f"truncating labels to first {n_logits} for partial evaluation.")
    labels = labels[:n_logits]

if probs.shape[0] != labels.shape[0]:
    raise ValueError(
        f"Logits count ({probs.shape[0]}) does not match labels count ({labels.shape[0]}). "
        "Check the dataset order or preprocessing."
    )

pred_indices = torch.argmax(probs, dim=1)
n_show = min(10, pred_indices.shape[0])
first5_pred_indices = pred_indices[:n_show].tolist()
first5_pred_names = [idx_to_class[i] for i in first5_pred_indices]
first5_gt_indices = labels[:n_show].tolist()
first5_gt_names = [idx_to_class[i] for i in first5_gt_indices]
print(f"First {n_show} predictions (class idx):", first5_pred_indices)
print(f"First {n_show} predictions (class name):", first5_pred_names)
print(f"First {n_show} ground truth (class idx):", first5_gt_indices)
print(f"First {n_show} ground truth (class name):", first5_gt_names)
for i in range(n_show):
    print(
        f"[{i}] pred={first5_pred_indices[i]} ({first5_pred_names[i]}) "
        f"gt={first5_gt_indices[i]} ({first5_gt_names[i]})"
    )

def topk_accuracy(preds: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)):
    maxk = max(topk)
    _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / preds.size(0)) * 100.0)
    return res

acc1, acc5 = topk_accuracy(probs, labels, topk=(1, 5))
print(f"Top-1 Accuracy: {acc1.item():.2f}%")
print(f"Top-5 Accuracy: {acc5.item():.2f}%")
