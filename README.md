<h1>🚀 Tiny Object Detection for Edge Devices</h1>

A lightweight tiny-object detection project designed for edge devices, starting with a synthetic dataset pipeline.
The goal is to build a hybrid training system by combining synthetic + real-world micro-object data, enabling robust detection even on extremely resource-constrained hardware.

---
*This repository currently includes a fully functional end-to-end synthetic data + YOLOv8 training pipeline running entirely on CPU.*
---

**✔ Implemented Features**

**✅ 1. Synthetic Dataset Generator**

A Python script (gen_dataset.py) that creates a complete YOLO-ready dataset:

Random 640×480 backgrounds

1–3 synthetic “vehicle-like” shapes

Accurate bounding box labels (YOLO format)

Configurable:

number of images

object count

object scale

image size

Allows training without using any real data (for now).

**✅ 2. Automated Train/Val Split**

Script: split_dataset.py

Generates:
```
data/synthetic/train/images
data/synthetic/train/labels
data/synthetic/val/images
data/synthetic/val/labels
```

Ensures label/image consistency

Automatically moves files into the correct YOLO structure

**✅ 3. YOLOv8 Training Pipeline (Working on CPU)**
```
Script: train.py
```
*Includes:*

yolov8n.pt (Nano model)

5-epoch training for synthetic verification

Batch preview images

Loss and metrics plotting

*Produces:*
```
runs/detect/train/weights/best.pt

runs/detect/train/weights/last.pt
```
Confirms the dataset + config + YOLO pipeline are correct.

**✅ 4. Dataset Configuration File**
```
configs/dataset.yaml

path: data/synthetic
train: train/images
val: val/images
names:
  0: object
```

Ready for YOLOv8 / YOLOv10 / other frameworks.

**✅ 5. Clean Project Structure**

Organized for clarity and future hybrid expansion:
```
tiny-object-det-edge-Hybrid/
│
├── scripts/
│   ├── gen_dataset.py
│   ├── split_dataset.py
│   └── train.py
│
├── configs/
│   └── dataset.yaml
│
├── data/
│   └── synthetic/
│       ├── train/
│       └── val/
│
├── runs/
│   └── detect/
│
└── README.md
```
**✅ 6. Verified Training Results**

The synthetic pipeline is fully validated:

Images load correctly

YOLO labels are correct

Batch preview images generated

YOLOv8 successfully detects tiny objects in validation

Training completes without errors

This establishes a stable foundation before adding real-world images.

---

**🎯 Project Goals**

Build a pipeline that supports both synthetic and real-world datasets

Target edge devices (Jetson Nano, Raspberry Pi, microcontrollers, etc.)

Design highly optimized training for tiny object detection

Enable domain randomization and simulation-to-real transfer

Support ONNX / TensorRT deployment later

---

**▶ How to Run the Project**
1️⃣ Install Dependencies
```
pip install ultralytics opencv-python numpy matplotlib
```

2️⃣ Generate Synthetic Dataset
```
python scripts/gen_dataset.py
```

3️⃣ Split into Train/Val
```
python scripts/split_dataset.py
```

4️⃣ Train YOLOv8 (CPU)
```
python scripts/train.py
```

Training results will be saved under runs/detect/train/.
