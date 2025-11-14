#!/usr/bin/env python3
import argparse
from ultralytics import YOLO

def main(data, model, epochs, imgsz, device):
    # model can be yolov8n.pt / yolov8s.pt or a custom .pt
    yolo = YOLO(model)
    # train
    yolo.train(data=data, epochs=epochs, imgsz=imgsz, device=device, batch=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu or 0 for gpu")
    args = parser.parse_args()
    main(args.data, args.model, args.epochs, args.imgsz, args.device)
