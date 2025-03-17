import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

def de_normalize(inp: torch.Tensor, mean = np.array([0.485, 0.456, 0.406]), std = np.array([0.229, 0.224, 0.225])):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) * 255
    return inp.astype(np.uint8)

def imsave(inputs: torch.Tensor, save_path):
    inp = make_grid(inputs, nrow=5)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig = plt.figure(figsize=(19,8))
    plt.imshow(inp)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def letter_box(img, imgsz=1024, padding_value=0):
    """
    将图像最大边缩放到指定大小, 并填充到指定大小
    """
    h, w = img.shape[:2]
    # 计算缩放比例
    scale = imgsz / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 缩放
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 计算填充尺寸
    dh = imgsz - new_h
    dw = imgsz - new_w
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    
    # 填充图像到指定大小
    new_img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    return new_img

def blend(img, mask):
    """
    将mask叠加到原图上
    """
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_HSV)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlap = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    overlap[mask == 0] = img[mask == 0]
    return overlap

def inverse_transform(pred: torch.Tensor, h, w, imgsz=1024):
    """
    将预测的mask还原到原图大小
    """
    scale = imgsz / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    # 计算填充尺寸
    dh = imgsz - new_h
    dw = imgsz - new_w
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    # 裁剪和缩放
    pred = pred[top:top+new_h, left:left+new_w]
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def show_mask(mask, ax, color=None, random_color=False):
    if color is None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, color=None):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green' if color is None else color.tolist(), facecolor=(0,0,0,0), lw=2))    

def get_color_map(normalized=False, alpha=0.5):
    # https://www.uied.cn/38589.html
    color_maps = np.array([
        [180, 120, 120], # 建筑
        [6, 230, 230],   # 天空
        [80, 50, 50],    # 地板
        [4, 200, 3],     # 树
        [120, 120, 80],  # 天花板
        [255, 9, 224],   # 房子
        [230, 230, 230], # 窗户
        [0, 102, 200],   # 汽车
        [4, 250, 7],     # 草
        [235, 255, 7],   # 人行道
        [150, 5, 61],    # 人
        [255, 6, 82],    # 桌子
        [204, 70, 32],   # 椅子
        [204, 5, 255],   # 床
        [143, 255, 140], # 山
        [120, 120, 120], # 墙
    ]) # (14, 3)
    if normalized:
        color_maps = color_maps.astype(float) / 255.0
    alpha = np.array([alpha]).reshape(-1,1).repeat(color_maps.shape[0], axis=0)
    color_maps = np.concatenate([color_maps, alpha], axis=1)
    return color_maps


def get_mem_desc():
    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
    return mem
