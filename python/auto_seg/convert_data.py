import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm

def convert_labelme_to_npy(images_dir, jsons_dir, output_dir, imgsz, padding_value=0):
    """将图像和labelme格式的分割标注转换为npy格式,先缩放到最大边长等于指定大小,然后再填充
    
    Args:
        images_dir: 图像目录路径
        jsons_dir: labelme json标注文件目录路径 
        output_dir: 输出npy文件的目录路径
        imgsz: 输出图像尺寸
        padding_value: 填充值,默认为0
    """
    images_dir = Path(images_dir)
    jsons_dir = Path(jsons_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "imgs").mkdir(exist_ok=True)
    (output_dir / "gts").mkdir(exist_ok=True)

    
    # 遍历所有图像文件
    image_files = list(images_dir.glob("*"))
    for img_path in tqdm(image_files):
        # 读取图像
        img = np.array(Image.open(img_path))
        
        # 归一化图像到[0,1]
        # img = img.astype(np.float32) / 255.0

        # 创建掩码图像
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 读取对应的json标注文件
        json_path = jsons_dir / f"{img_path.stem}.json"
        if not json_path.exists():
            continue
            
        with open(json_path, encoding='utf-8') as f:
            label_data = json.load(f)
    
        # 解析所有多边形标注
        for i, shape in enumerate(label_data["shapes"], start=1):
            if shape["shape_type"] != "polygon":
                continue
                
            # 将多边形点转换为numpy数组
            points = np.array(shape["points"], dtype=np.int32)
            
            # 绘制填充多边形,使用标签值填充
            cv2.fillPoly(mask, [points], 255)

        # # 计算缩放比例
        # scale = imgsz / max(h, w)
        # new_h = int(h * scale)
        # new_w = int(w * scale)
        
        # # 缩放图像和掩码
        # img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # # 计算填充尺寸
        # dh = imgsz - new_h
        # dw = imgsz - new_w
        # top = dh // 2
        # bottom = dh - top
        # left = dw // 2
        # right = dw - left
        
        # # 填充图像和掩码到指定大小
        # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
        # mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            
        # 保存为npy格式
        # np.save(str(output_dir / "imgs" / f"{img_path.stem}.png"), img)
        cv2.imwrite(str(output_dir / "imgs" / f"{img_path.stem}.png"), img)
        cv2.imwrite(str(output_dir / "gts" / f"{img_path.stem}.png"), mask)

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--images_dir", type=str, required=True, help="图像目录路径")
    parser.add_argument('-j', "--jsons_dir", type=str, required=True, help="labelme json标注文件目录路径")
    parser.add_argument('-o', "--output_dir", type=str, required=True, help="输出npy文件的目录路径")
    parser.add_argument('--imgsz', type=int, default=1024, help='图像尺寸')
    args = parser.parse_args()
    
    convert_labelme_to_npy(args.images_dir, args.jsons_dir, args.output_dir, args.imgsz)