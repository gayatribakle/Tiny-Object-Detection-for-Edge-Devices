#!/usr/bin/env python3
import os, shutil, argparse, random

def mkdir(p): 
    os.makedirs(p, exist_ok=True)

def main(src, train_ratio=0.8):
    imgs = sorted(os.listdir(os.path.join(src, "images")))
    random.shuffle(imgs)
    ntrain = int(len(imgs) * train_ratio)
    train = imgs[:ntrain]
    val = imgs[ntrain:]

    for subset, names in [("train", train), ("val", val)]:
        img_out = os.path.join(src, subset, "images")
        lbl_out = os.path.join(src, subset, "labels")
        mkdir(img_out); mkdir(lbl_out)
        for name in names:
            shutil.copy(os.path.join(src, "images", name),
                        os.path.join(img_out, name))
            base = os.path.splitext(name)[0] + ".txt"
            shutil.copy(os.path.join(src, "labels", base),
                        os.path.join(lbl_out, base))
    print("Split complete. Train:", len(train), "Val:", len(val))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/synthetic", help="src dataset root")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()
    main(args.src, args.train_ratio)
