#!/usr/bin/env python3
"""
Generate synthetic dataset for tiny object detection (YOLO format).
Creates random rectangles (vehicles) on random background colors, with varying scales.
"""

import os
import random
import argparse
from PIL import Image, ImageDraw
import json
import math

def make_dirs(p):
    os.makedirs(p, exist_ok=True)

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def generate_image(w, h, n_objs, min_scale=0.01, max_scale=0.06):
    img = Image.new("RGB", (w, h), random_color())
    draw = ImageDraw.Draw(img)
    labels = []
    for _ in range(n_objs):
        # choose tiny box size as fraction of image
        scale = random.uniform(min_scale, max_scale)
        box_w = int(scale * w)
        box_h = int(scale * h * random.uniform(0.5,1.2))
        x1 = random.randint(0, max(0, w - box_w))
        y1 = random.randint(0, max(0, h - box_h))
        x2 = x1 + box_w
        y2 = y1 + box_h
        # draw a simple vehicle-like rectangle with roof
        draw.rectangle([x1, y1 + box_h//3, x2, y2], fill=random_color())
        # small roof
        draw.polygon([(x1 + box_w*0.2, y1 + box_h*0.3),
                      (x1 + box_w*0.8, y1 + box_h*0.3),
                      (x1 + box_w*0.6, y1),
                      (x1 + box_w*0.4, y1)], fill=random_color())
        # YOLO format: class x_center y_center width height (normalized)
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        labels.append((0, x_center, y_center, bw, bh))
    return img, labels

def save_yolo_label(path, labels):
    with open(path, "w") as f:
        for l in labels:
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*l))

def main(out_dir, n_images, img_w, img_h, max_objs):
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    make_dirs(images_dir)
    make_dirs(labels_dir)
    for i in range(n_images):
        n_objs = random.randint(1, max_objs)
        img, labels = generate_image(img_w, img_h, n_objs)
        img_name = f"img_{i:04d}.jpg"
        label_name = f"img_{i:04d}.txt"
        img.save(os.path.join(images_dir, img_name))
        save_yolo_label(os.path.join(labels_dir, label_name), labels)
        if i % 50 == 0:
            print(f"Generated {i}/{n_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic", help="output dataset root")
    parser.add_argument("--n_images", type=int, default=200, help="number images")
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    parser.add_argument("--max_objs", type=int, default=3)
    args = parser.parse_args()
    main(args.out, args.n_images, args.img_w, args.img_h, args.max_objs)
