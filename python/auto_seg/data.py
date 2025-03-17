import random
import json
from pathlib import Path

from typing import Union, List, Tuple, Type

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import cv2
import numpy as np
from PIL import Image

from utils import letter_box

class FewShotDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        target_dir: Path,
        n: int,
    ):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = T.Compose([
            # T.Resize((1024, 1024), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_list = list(self.image_dir.iterdir())
        if n > 0:
            image_list = random.sample(image_list, n)
        self.image_list = self.filter_by_mask(image_list)

    def filter_by_mask(self, image_list):
        new_image_list = []
        for image_path in image_list:
            mask_path = self.target_dir / (image_path.stem + '.png')
            if mask_path.exists():
                new_image_list.append(image_path)
        return new_image_list

    def __len__(self):
        return len(self.image_list)

    def preprocess(self, img, mask, imgsz=1024):
        return letter_box(img, imgsz), letter_box(mask, imgsz)


    def __getitem__(self, index):
        img_path = self.image_list[index]
        image = Image.open(str(img_path)).convert('RGB')
        image_np = np.array(image)
        
        mask_path = self.target_dir / (img_path.stem + '.png')
        gt = Image.open(str(mask_path))
        mask = np.array(gt)
        mask[mask>0] = 1
        image_np, mask = self.preprocess(image_np, mask)
        image_tensor = self.transform(image_np)
        return {
            'img': image_tensor, # (3, 1024, 1024)
            'mask': torch.tensor(mask[None, :, :]).long(), # (1, 1024, 1024)
        }

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img' or k == 'mask':
                value = torch.stack(value, 0)
            # if k in ['masks', 'keypoints', 'bboxes', 'cls', 'valid_segments']:
            #     value = torch.cat(value, 0)
            new_batch[k] = value
        return new_batch


def build_dataloader(image_dir, target_dir, n, batch_size, num_workers=4):
    dataset = FewShotDataset(image_dir, target_dir, n)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


class LabelMeDataset(Dataset):
    def __init__(self, image_dir: Path, target_dir: Path, classes=None):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.classes = classes

        image_list = list(self.image_dir.iterdir())
        if classes is not None:
            image_list = self.filter_by_classes(image_list, classes)
        self.image_list = image_list
            
    
    def filter_by_classes(self, image_list, classes):
        new_image_list = []
        for image_path in image_list:
            json_path = self.target_dir / (image_path.stem + '.json')
            if not json_path.exists():
                continue
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data['shapes']:
                    if shape['shape_type'] == 'rectangle' and shape['label'] in classes:
                        new_image_list.append(image_path)
                        break
        return new_image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(str(image_path)).convert('RGB')
        image_np = np.array(image)
        json_path = self.target_dir / (image_path.stem + '.json')
        boxes = []
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data['shapes']:
                    if shape['shape_type'] == 'rectangle':
                        if self.classes is not None and shape['label'] not in self.classes:
                            continue
                        points = shape['points']
                        x0, y0 = points[0][0], points[0][1]
                        x1, y1 = points[1][0], points[1][1]
                        boxes.append([x0, y0, x1, y1])
        return {
            "img": image_np,
            "box": np.array(boxes),
            'im_file': image_path,
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            # if k == 'img' or k == 'mask':
            #     value = torch.stack(value, 0)
            # if k in ['masks', 'keypoints', 'bboxes', 'cls', 'valid_segments']:
            #     value = torch.cat(value, 0)
            new_batch[k] = value
        return new_batch


def build_labelme_dataloader(image_dir, target_dir, classes=None, batch_size=8, num_workers=4):
    dataset = LabelMeDataset(image_dir, target_dir, classes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=getattr(dataset, 'collate_fn', None)
    )
    return dataloader