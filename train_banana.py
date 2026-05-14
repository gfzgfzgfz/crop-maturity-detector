#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Banana Ripeness Classification - YOLOv8 Training"""

from ultralytics import YOLO
import os
import multiprocessing

DATA_DIR = r"F:\系统\yolo-u\ultralytics\datasets\Banana Ripeness Classification.v1-original-images.folder"
OUTPUT_DIR = r"F:\系统\yolo-u\runs\classify\banana_ripeness"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Banana Ripeness Classification Training")
    print(f"  Data: {DATA_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    model = YOLO('yolov8n-cls.pt')

    results = model.train(
        data=DATA_DIR,
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda',
        name='banana_ripeness',
        project=r"F:\系统\yolo-u\runs\classify",
        exist_ok=True,
        patience=15,
        augment=True,
        amp=False,
        workers=0,          # single-process to avoid Windows multiprocessing hang
        resume=True,        # resume from last.pt
    )

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best model: {OUTPUT_DIR}/weights/best.pt")
    print(f"  Last model: {OUTPUT_DIR}/weights/last.pt")
    print("  Usage: model = YOLO(r'" + OUTPUT_DIR + r"\weights\best.pt')")
    print("=" * 60)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
