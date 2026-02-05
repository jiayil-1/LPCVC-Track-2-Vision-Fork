import qai_hub
import onnx
import os
import sys
import numpy as np
from compile_and_profile import compile_model
# --- Configuration ---
ONNX_DIR = "exported_onnx"
VIDEO_ONNX_NAME = "r2plus1dQEVD"   # change if you named it differently
DEVICE_NAME = "Snapdragon 8 Elite QRD"

# Must match your export dummy input
BATCH = 1
C = 3
T = 8           # frames (8 if that’s what you exported)
H = 112         # common for r2plus1d_18
W = 112
# ---------------------

data_path = "./preprocessed/test"

def inference_job(model, device, dataset):
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs = dataset
    )
    output = inference_job.download_output_data()
    return inference_job.job_id

video_tensors = []
labels = []
class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(data_path)))}
print(class_to_idx)
for cls in class_to_idx:
    cls_dir = os.path.join(data_path, cls)
    for fname in os.listdir(cls_dir):
        if not fname.endswith(".npy"):
            continue
        x = np.load(os.path.join(cls_dir, fname))  # (1,3,T,H,W)
        video_tensors.append(x.astype(np.float32))
        labels.append(class_to_idx[cls])

print("Loaded", len(video_tensors), "samples")
device = qai_hub.Device(DEVICE_NAME)
VIDEO_ONNX_PATH = os.path.join(ONNX_DIR, VIDEO_ONNX_NAME)
if not os.path.exists(VIDEO_ONNX_PATH):
    print(f"Error: '{VIDEO_ONNX_PATH}' not found. Run export_onnx.py first.")
    sys.exit(1)

print(f"Loading ONNX video model from {VIDEO_ONNX_PATH}...")
onnx_video_model = onnx.load(VIDEO_ONNX_PATH)

try:
    onnx.checker.check_model(onnx_video_model)
    print("Video ONNX model is valid ✅")
except onnx.checker.ValidationError as e:
    print("Video ONNX model validation failed ❌")
    print(e)
    sys.exit(1)

input_specs = {
    "video": (BATCH, C, T, H, W)  # float32 by default
}

compile_job = compile_model(
    model=onnx_video_model,
    device=device,
    input_specs=input_specs,
)
target_model = qai_hub.get_job(compile_job).get_target_model()
dataset = qai_hub.upload_dataset(
    {"video": video_tensors},
    name="test_set"
)
inference_id = inference_job(
    model=target_model,
    device=device,
    dataset = dataset
)
print(f"Compilation job ID: {inference_id}")
print("Done")