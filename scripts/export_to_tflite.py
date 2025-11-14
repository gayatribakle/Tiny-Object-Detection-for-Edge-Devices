#!/usr/bin/env python3
"""
Export YOLO weights -> ONNX -> TFLite (float16 or int8)
Requires: onnx, onnxruntime, tensorflow
"""

import argparse
import os
import subprocess
from ultralytics import YOLO
import tensorflow as tf
import numpy as np

def export_onnx(weights, onnx_path="model.onnx"):
    model = YOLO(weights)
    # ultralytics has .export
    model.export(format="onnx", imgsz=640, simplify=True)
    # ultralytics will save to runs/val/exp/weights/xxx.onnx by default; find it
    # We'll assume it saved model.onnx in current folder
    print("ONNX export attempted. Check working directory for .onnx file.")

def convert_onnx_to_tflite(onnx_path, tflite_path, quantize=None):
    # Use tf.experimental.onnx? Not available. We'll use ONNX runtime to load and TensorFlow to convert via a tf.saved_model conversion is complex.
    # Alternative: use tf2onnx or onnx-tf. Keep a minimal fallback: convert using tf-onnx if available.
    try:
        import onnx
        import onnx_tf
    except Exception as e:
        print("For full ONNX->TFLite conversion, install onnx and onnx-tf or use an external converter. Exiting.")
        return
    # This script includes a minimal approach; users often convert ONNX->SavedModel using onnx-tf, then SavedModel->TFLite.
    from onnx_tf.backend import prepare
    model = onnx.load(onnx_path)
    tf_rep = prepare(model)
    sm_dir = "temp_saved_model"
    tf_rep.export_graph(sm_dir)
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For full int8, representative dataset is required. Provide simple synthetic generator here.
        def repr_dataset():
            for _ in range(100):
                yield [np.random.rand(1,640,640,3).astype(np.float32)]
        converter.representative_dataset = repr_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Converted TFLite:", tflite_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="models/model.tflite")
    parser.add_argument("--quantize", choices=[None, "float16", "int8"], default=None)
    args = parser.parse_args()
    export_onnx(args.weights, onnx_path="model.onnx")
    convert_onnx_to_tflite("model.onnx", args.output, quantize=args.quantize)
