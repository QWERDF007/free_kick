import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import monai
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms as T

sam_root = Path(os.getcwd()).parent.parent.parent / '3rdparty' 
if (sam_root / 'segment_anything').exists():
    sys.path.append(str(sam_root))
sam_root, sam_root.exists()

from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor

from model import FewShotSAM
from data import build_dataloader
from utils import imsave, letter_box, blend, inverse_transform, get_mem_desc
from gen_mask_from_box import gen_mask_from_box

parser = argparse.ArgumentParser()
parser.add_argument(
    "-train",
    "--train_path",
    type=str,
    default="data/npy/CT_Abd",
    help="path to training files; two subfolders: gts and imgs",
)
parser.add_argument(
    "-test",
    "--test_path",
    type=str,
    default="",
    help="path to testing files; two subfolders: gts and imgs",
)
parser.add_argument('-data_type', type=str, default='mask', choices=['mask', 'box'])
parser.add_argument('-c', '--classes', type=str, default=None, help='classes to be segmented')
parser.add_argument("-task_name", type=str, default="FewShotSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-ckpt", "--checkpoint", type=str, default="checkpoints/SAM/sam_vit_b_01ec64.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=20)
parser.add_argument('-b', "--batch_size", type=int, default=4)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument('-n', "--fewshot", type=int, default=-1)
parser.add_argument('-imgsz', "--imgsz", type=int, default=1024)
args = parser.parse_args()

def train_from_mask(args):
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    fewshot_model = FewShotSAM(image_encoder=sam_model.image_encoder)
    fewshot_model.train()

    print("load mask decoder weight from sam model")
    # 在加载参数前过滤不匹配的键
    pretrained_dict = sam_model.mask_decoder.state_dict()
    model_dict = fewshot_model.mask_decoder.state_dict()
    # 过滤条件：键名存在且形状匹配
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.size() == model_dict[k].size()
    }
    # 加载过滤后的参数
    fewshot_model.mask_decoder.load_state_dict(filtered_dict, strict=False)

    fewshot_model.to(args.device)
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in fewshot_model.parameters()),
    ) 
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in fewshot_model.parameters() if p.requires_grad),
    ) 
    optimizer = torch.optim.AdamW(
        fewshot_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # loss
    dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    work_dir = Path(args.work_dir)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_dir = work_dir / (args.task_name + "-" + run_id)

    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.train_path)
    image_dir = train_dir / 'imgs'
    target_dir = train_dir / 'gts'
    train_loader = build_dataloader(image_dir, target_dir, args.fewshot, args.batch_size, args.num_workers)
    
    steps_per_epoch = len(train_loader)
    print("Number of training samples: ", len(train_loader.dataset))
    print("Number of iterations per epoch: ", steps_per_epoch)

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # 生成 fewshot target embeddings
    images = []
    targets = []
    for batch in train_loader:
        images.append(batch['img'])
        targets.append(batch['mask'])
      
    images = torch.cat(images, dim=0)
    imsave(images, str(model_save_dir / 'fewshot_samples.png'))
    targets = torch.cat(targets, dim=0)
    print('images', images.shape, 'targets', targets.shape)
    fewshot_target_embeddings = []
    pbar = tqdm(range(images.shape[0]))
    pbar.set_description(f'Generating fewshot target embeddings, mem: {get_mem_desc()}')
    for i in pbar:
        fewshot_target_embeddings.append(fewshot_model.get_fewshot_target_embeddings(images[i].unsqueeze(0).to(args.device), targets[i].unsqueeze(0).to(args.device))) # (1, 256)
        pbar.set_description(f'Generating fewshot target embeddings, mem: {get_mem_desc()}')
    fewshot_target_embeddings = torch.cat(fewshot_target_embeddings, dim=0)
    np.save(str(model_save_dir / 'fewshot_target_embeddings.npy'), fewshot_target_embeddings.cpu().numpy())
    
    # 微调
    iter_num = 1
    total_iter_num = args.num_epochs * steps_per_epoch
    total_loss = 0
    best_loss = 1e10
    pbar = tqdm(range(start_epoch, args.num_epochs))
    for epoch in pbar:
        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_ce_loss = 0
        for batch in train_loader:
            image = batch['img'].to(args.device)
            target = batch['mask'].to(args.device)
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    preds = fewshot_model(image, fewshot_target_embeddings.expand(image.shape[0], -1, -1))
                    dice_loss_value = dice_loss(preds, target)
                    ce_loss_value = ce_loss(preds, target.float())
                    loss = dice_loss_value + ce_loss_value
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                preds = fewshot_model(image, fewshot_target_embeddings.expand(image.shape[0], -1, -1))
                dice_loss_value = dice_loss(preds, target)
                ce_loss_value = ce_loss(preds, target.float())
                loss = dice_loss_value + ce_loss_value
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            epoch_loss += loss.item()
            epoch_dice_loss += dice_loss_value.item()
            epoch_ce_loss += ce_loss_value.item()
            iter_num += 1
        epoch_loss /= steps_per_epoch
        epoch_dice_loss /= steps_per_epoch
        epoch_ce_loss /= steps_per_epoch
        
        pbar.set_description(
                f'Epoch: {epoch:03d}/{args.num_epochs:03d}, Step: {iter_num:04d}/{total_iter_num:04d}, '
                f'dice_loss: {epoch_dice_loss:.4f}, ce_loss: {epoch_ce_loss:.4f}, loss: {epoch_loss:.4f}, '
                f'loss_avg: {total_loss/iter_num:.4f}, mem: {get_mem_desc()}')

        # 保存模型
        checkpoint = {
            "model": fewshot_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, str(model_save_dir / "fewshot_model_latest.pth"))
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": fewshot_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, str(model_save_dir / "fewshot_model_best.pth"))

    return model_save_dir, fewshot_model

def train_from_box(args):
    # 加载 SAM 模型
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam_model.to(device=args.device)
    predictor = SamPredictor(sam_model)
    image_dir = Path(args.train_path) / 'imgs'
    data_dir = Path(args.train_path) / 'labels'
    out_dir = Path(args.train_path) / 'gts'
    # 生成并挑选微调的 mask
    gen_mask_from_box(predictor, image_dir, data_dir, out_dir, args.classes, args.imgsz)
    del predictor
    del sam_model
    torch.cuda.empty_cache()
    # 微调模型
    model_save_dir, fewshot_model = train_from_mask(args)
    # shutil.rmtree(out_dir)
    return model_save_dir, fewshot_model

    

def test(args, model_save_dir, fewshot_model):
    # 加载 embedding
    fewshot_target_embeddings = np.load(str(model_save_dir / 'fewshot_target_embeddings.npy'))
    fewshot_target_embeddings = torch.from_numpy(fewshot_target_embeddings).to(args.device)
    # 加载模型
    best_model_path = model_save_dir / 'fewshot_model_best.pth'
    print("Loading best model from ", best_model_path)
    checkpoint = torch.load(best_model_path, map_location='cpu')
    fewshot_model.load_state_dict(checkpoint['model'])
    fewshot_model.eval()

    # 创建输出目录
    print("Creating output directory...")
    test_dir = Path(args.test_path)
    test_out = model_save_dir / 'out'
    if not test_out.exists():
        test_out.mkdir(parents=True, exist_ok=True)


    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_image_dir = test_dir / 'imgs'
    test_target_dir = test_dir / 'gts'
    test_label_dir = test_dir / 'labels'
    test_images = list(test_image_dir.iterdir())
    pbar = tqdm(test_images)
    pbar.set_description(f'Testing..., mem: {get_mem_desc()}')
    for image_path in pbar:
        pbar.set_description(f'Testing..., mem: {get_mem_desc()}')
        image = Image.open(str(image_path)).convert('RGB') # (H, W, 3)
        image_np = np.array(image) # (H, W, 3)
        mask_path = test_target_dir / (image_path.stem + '.png')
        if not mask_path.exists():
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            mask = np.array(Image.open(str(mask_path)).convert('L')) # (H, W)
        image_tensor = test_transform(letter_box(image_np)).unsqueeze(0).to(args.device) # (1, 3, 1024, 1024)
        pred = fewshot_model.predict(image_tensor, fewshot_target_embeddings.expand(image_tensor.shape[0], -1, -1)) # (B, 1, 1024, 1024)
        pred = pred.squeeze().cpu().numpy() # np.bool [True, False]
        pred = inverse_transform(pred.astype(np.uint8), image.height, image.width, args.imgsz) # (H, W)
        pred[pred > 0] = 255
        mask_blend = blend(image_np, mask)
        label_path = test_label_dir / (image_path.stem + '.json')
        if label_path.exists():
            with open(str(label_path), 'r', encoding='utf-8') as f:
                label = json.load(f)
                for shape in label['shapes']:
                    if shape['label'] in args.classes:
                        x0, y0 = int(shape['points'][0][0]), int(shape['points'][0][1])
                        x1, y1 = int(shape['points'][1][0]), int(shape['points'][1][1])
                        cv2.rectangle(mask_blend, (x0, y0), (x1, y1), (255, 0, 0), 2)
        pred_blend = blend(image_np, pred)
        show = np.hstack([image_np, mask_blend, pred_blend])
        plt.figure(figsize=(19,8))
        plt.imshow(show)
        plt.axis('off')
        plt.savefig(str(test_out / (image_path.stem + '.png')), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


if __name__ == "__main__":
    
    if args.data_type == 'mask':
        model_save_dir, fewshot_model = train_from_mask(args)
    elif args.data_type == 'box':
        model_save_dir, fewshot_model = train_from_box(args)
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")
    if args.test_path is None or not Path(args.test_path).exists():
        print("No test data found, skip testing")
    else:
        test(args, model_save_dir, fewshot_model)
