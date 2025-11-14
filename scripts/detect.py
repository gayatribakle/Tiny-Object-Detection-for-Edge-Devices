#!/usr/bin/env python3
import argparse, os
from ultralytics import YOLO

def main(weights, source, conf=0.25, save_dir="runs/detect"):
    model = YOLO(weights)
    res = model.predict(source=source, conf=conf, save=True, save_txt=True, project="runs/detect", name="exp")
    print("Saved inference results to runs/detect/exp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="data/sample_inputs")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    main(args.weights, args.source, args.conf)
