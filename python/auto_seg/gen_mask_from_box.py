import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sam_root = Path(os.getcwd()).parent.parent.parent / '3rdparty' 
if (sam_root / 'segment_anything').exists():
    sys.path.append(str(sam_root))
sam_root, sam_root.exists()

from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor

import matplotlib.pyplot as plt
from data import build_labelme_dataloader
from utils import show_mask, show_points, show_box, get_color_map, get_mem_desc

save_flag = False
finish_flag = False

def on_key_press(event):
    """
    键盘事件处理, w 存图, q 退出, esc 结束
    """
    global save_flag, finish_flag
    if event.key == 'w':
        save_flag = True
        plt.close()
    elif event.key == 'escape':
        save_flag = False
        finish_flag = True
        plt.close()
    elif event.key == 'q':
        save_flag = False
        plt.close()


def gen_mask_from_box(predictor, image_dir : Path, data_dir : Path, out_dir : Path, classes=None, imgsz=1024):
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    dataloader = build_labelme_dataloader(image_dir, data_dir, classes, batch_size=1)
    color_maps = get_color_map(normalized=True)
    for batch in dataloader:
        images = batch['img']
        boxes = batch['box']
        im_files = batch['im_file']
        for image, box, im_file in zip(images, boxes, im_files):
            print(im_file, box.shape, 'mem:', get_mem_desc())
            predictor.set_image(image)
            # 框转换成 Tensor
            input_boxes = torch.tensor(box, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            # 批量预测
            masks, scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            # mask.shape # (B, num_predicted_masks_per_input, H, W)
            fig = plt.figure(figsize=(19, 8))
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            plt.imshow(image)
            mask_np = np.zeros((masks.shape[2], masks.shape[3]), np.uint8)
            for i, mask in enumerate(masks):
                mask = mask.cpu().squeeze().numpy() # (H, W), bool
                mask_np[mask] = 255
                show_mask(mask, plt.gca(), color=color_maps[i]) 
            for i, box in enumerate(input_boxes):
                show_box(box.cpu().numpy(), plt.gca(), color=color_maps[i])

            plt.axis('off')
            plt.show()
            # 保存 mask
            if save_flag:
                mask_save_path = out_dir / (im_file.stem + '.png')
                Image.fromarray(mask_np).save(str(mask_save_path))
            if finish_flag:
                return
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', type=str, default='vit_b')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default='checkpoints/SAM/sam_vit_b_01ec64.pth')
    parser.add_argument('-i', '--image_dir', type=str, default='data/images')
    parser.add_argument('-d', '--data_dir', type=str, default='data/labels')
    parser.add_argument('-c', '--classes', type=str, default=None)
    parser.add_argument('-o', '--out_dir', type=str, default='data/masks')
    parser.add_argument('-imgsz', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam_model.to(device=args.device)
    predictor = SamPredictor(sam_model)
    gen_mask_from_box(predictor, args.image_dir, args.data_dir, args.out_dir, args.classes, args.imgsz)

