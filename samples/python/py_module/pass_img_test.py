# import cv2
# import torch
# from ultralytics import YOLO

# def pass_img_test(img):
#     model = YOLO("F:/models/yolov8/yolov8s.pt")
#     results = model(img)
#     print(results, flush=True)
#     return img

import sys
import os
# print('sys.path:', flush=True)
# print(sys.path, flush=True)
# print('PATH:', flush=True)
# print(os.environ['PATH'], flush=True)

from ultralytics import YOLO
import numpy


class Yolov8Detection:
    def __init__(self, model_path, imgsz, device):
        
        print("Yolov8Detection init", flush=True)
        self.init_model(model_path, imgsz, device)
    
    def init_model(self, model_path, imgsz, device):
        print("Yolov8Detection init_model", flush=True)
        self.model_path = model_path
        self.imgsz = imgsz
        self.device = device
        # self.model = YOLO(self.model_path)

    def detect(self, img):
        print("Yolov8Detection detect", flush=True)
        # result = self.model(img)
        # print(result, flush=True)