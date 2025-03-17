from typing import Union, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.mask_decoder import MLP
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
from segment_anything.modeling.transformer import TwoWayTransformer

image_size = 1024
vit_patch_size = 16
prompt_embed_dim = 256
# 64
image_embedding_size = image_size // vit_patch_size 


class FewShotTargetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)


    def forward(
            self, 
            image_embeddings: torch.Tensor, # (n, 256, 64, 64)
            binary_mask: Union[torch.Tensor, None], # (n, 1, 1024, 1024)
        ):
        if binary_mask is None:
            return self.avg(image_embeddings).reshape(image_embeddings.shape[0], -1) # (n, 256)
        # 下采样mask到image_embedding_size (64x64)
        mask = F.interpolate(
            binary_mask.float(),
            size=(image_embedding_size, image_embedding_size),
            mode='nearest'
        ) # (B, 1, 64, 64)
        zi = self.avg(image_embeddings * mask) / self.avg(mask) # (n, 256, 1, 1)
        return zi.reshape(zi.shape[0], -1) # (n, 256)


class FewShotMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 2,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim # 256
        self.transformer = transformer # TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)

        self.num_multimask_outputs = num_multimask_outputs # 3

        self.iou_token = nn.Embedding(1, transformer_dim)                      # 1x256
        self.num_mask_tokens = num_multimask_outputs + 1                       # 3 # fg, bg, extra only for multiple prompts
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim) # 4x256

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),      # in=256, out=64, 2x2, 2
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), # in=64, out=16, 2x2, 2
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,          # (B, 256, 64, 64)
        image_pe: torch.Tensor,                  # (1, 256, 64, 64)
        fewshot_target_embeddings: torch.Tensor, # (B, n, 256)
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # masks: (B, 4, 256, 256), iou_pred: (B, 4)
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,                 # (B, 256, 64, 64)
            image_pe=image_pe,                                 # (1, 256, 64, 64)
            fewshot_target_embeddings=fewshot_target_embeddings, # (B, n, 256)
        )

        # Select the correct mask or masks for output
        # 根据是否需要输出多个掩码来选择不同的切片
        if multimask_output:
            # 如果需要多个掩码,选择除第一个掩码外的所有掩码(索引1到末尾)
            mask_slice = slice(1, None)
        else:
            # 如果只需要单个掩码,只选择第一个掩码(索引0)
            mask_slice = slice(0, 1)
        
        # 根据切片选择对应的掩码和IoU预测结果
        masks = masks[:, mask_slice, :, :] # (B, 1, 256, 256) or (B, 3, 256, 256)
        iou_pred = iou_pred[:, mask_slice] # (B, 1) or (B, 3)

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,         # (B, 256, 64, 64)
        image_pe: torch.Tensor,                 # (1, 256, 64, 64)
        fewshot_target_embeddings: torch.Tensor, # (B, n, 256)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)           # [1x256, (2+1)x256] -> [4x256]
        output_tokens = output_tokens.unsqueeze(0).expand(fewshot_target_embeddings.size(0), -1, -1) # (B, 4, 256)
        tokens = torch.cat((output_tokens, fewshot_target_embeddings), dim=1)                        # (B, 4+n, 256)

        # 在批次维度上扩展每个图像的数据以匹配每个掩码
        # 如果图像嵌入的批次大小与token的批次大小不匹配
        if image_embeddings.shape[0] != tokens.shape[0]:
            # 使用repeat_interleave在批次维度上重复图像嵌入,使其与token的批次大小相同
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            # 如果批次大小已经匹配,直接使用图像嵌入
            src = image_embeddings
        # src = src + dense_prompt_embeddings # (B, 256, 64, 64)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) # (B, 256, 64, 64) # 位置编码
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)           # (B, 4+n, 256) , (B, 4096, 256)
        iou_token_out = hs[:, 0, :]                                # (B, 256)  
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] # (B, 3, 256)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)      # (B, 4096, 256) -> (B, 256, 64, 64)
        upscaled_embedding = self.output_upscaling(src) # (B, 32, 256, 256) 使用二维转置卷积上采样
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1) # (B, 3, 32)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) # (B, 3, 256, 256)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out) # (B, 3) masks 和 iou 的 dim 都和 num_mask_tokens 有关
        return masks, iou_pred


class FewShotSAM(nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = FewShotMaskDecoder(
            num_multimask_outputs=2,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.pe_layer = PositionEmbeddingRandom(prompt_embed_dim // 2)
        self.fewshot_target_encoder = FewShotTargetEncoder()

        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def get_dense_pe(self):
        return self.pe_layer((image_embedding_size, image_embedding_size)).unsqueeze(0)
  
    @torch.no_grad()
    def get_fewshot_target_embeddings(
            self, 
            image: torch.Tensor,  # (n, 3, 1024, 1024)
            binary_mask: torch.Tensor # (n, 1, 1024, 1024)
        ):
        image_embedding = self.image_encoder(image) # (n, 256, 64, 64)
        fewshot_target_embeddings = self.fewshot_target_encoder(image_embedding, binary_mask) # (n, 256)
        return fewshot_target_embeddings
        
        

    def forward(self, 
                image: torch.Tensor, # (B, 3, 1024, 1024)
                fewshot_target_embeddings: torch.Tensor, # (B, n, 256) , n 是fewshot的个数
        ):
        with torch.no_grad():
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
          
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.get_dense_pe(),
            fewshot_target_embeddings=fewshot_target_embeddings,
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks
        
    @torch.no_grad()
    def predict(
        self, 
        image: torch.Tensor,                     # (B, 3, 1024, 1024)
        fewshot_target_embeddings: torch.Tensor, # (B, n, 256)
        threshold: float = 0.5
    ):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        # fewshot_target_embeddings = fewshot_target_embeddings.expand(image.shape[0], -1, -1) # (B, n, 256)

        low_res_logits, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.get_dense_pe(),
            fewshot_target_embeddings=fewshot_target_embeddings,
            multimask_output=False,
        )
        low_res_pred = torch.sigmoid(low_res_logits)  # (B, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, 1024, 1024)
        low_res_masks = low_res_pred.gt(threshold)  # (B, 1, 1024, 1024)
        return low_res_masks


