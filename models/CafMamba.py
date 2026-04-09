"""
CAF-Mamba model implementation.
This module defines modules and blocks used in the CafMamba architecture.
The CAF-Mamba model implemented here is used for multimodal features, and LMVD dataset is recommended, please refer to the paper:
"CAF-Mamba: Mamba-Based Cross-Modal Adaptive Attention Fusion for Multimodal Depression Detection" Section 3.2.1.

---------
Author:Bowen Zhou
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.normalization import LayerNorm

# Mamba
from mamba_ssm import Mamba
from .base import BaseNet


class ResidualMambaLayer(nn.Module):
    """
    This class implements a Residual Mamba Layer.
    """

    def __init__(self, d_model, dropout=0.0, mamba_config=None):
        super().__init__()
        assert mamba_config != None

        self.mamba = Mamba(d_model=d_model, **mamba_config)

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out


class ResMambaBlock(nn.Module):
    """
    This class implements the Residual Mamba Block.

    Stacked Residual Mamba Layers can be used further, in CAF-Mamba we use only one ResMamba layer per ResMamaba block.
    """

    def __init__(self, output_sizes=[256], dropout=0.0, mamba_config=None):
        super().__init__()

        mamba_list = []
        for i in range(len(output_sizes)):
            mamba_list.append(
                ResidualMambaLayer(
                    d_model=output_sizes[i],
                    dropout=dropout,
                    mamba_config=mamba_config,
                )
            )

        self.mamba_layers = torch.nn.ModuleList(mamba_list)

    def forward(
        self,
        x,
        inference_params=None,
    ):
        out = x
        for mamba_layer in self.mamba_layers:
            out = mamba_layer(
                out,
                inference_params=inference_params,
            )

        return out


class UnimodalExtractionModule(nn.Module):
    """
    This class implements the Unimodal Feature Extraction Module.
    This module contains three Unimodal Feature Extraction (UFE) blocks, each for one modality.
    """

    def __init__(self, mm_output_sizes, dropout, mamba_config):
        super().__init__()
        self.UFE_a = ResMambaBlock(
            mm_output_sizes, dropout=dropout, mamba_config=mamba_config
        )

        self.UFE_lau = ResMambaBlock(
            mm_output_sizes, dropout=dropout, mamba_config=mamba_config
        )

        self.UFE_egh = ResMambaBlock(
            mm_output_sizes, dropout=dropout, mamba_config=mamba_config
        )

    def forward(self, xa, xv_lau, xv_egh):
        xa = self.UFE_a(xa)
        xv_lau = self.UFE_lau(xv_lau)
        xv_egh = self.UFE_egh(xv_egh)
        return xa, xv_lau, xv_egh


class ModalWiseAttention(nn.Module):
    """
    This class implements the Modal-wise Attention Block.
    """

    def __init__(self, mm_input_size):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.conv1 = nn.Conv1d(
            mm_input_size * 4, mm_input_size * 2, 1, padding=0, dilation=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            mm_input_size * 2, mm_input_size, 1, padding=0, dilation=1, bias=False
        )

    def forward(self, xa, xv_lau, xv_egh, xi):
        avg1 = torch.mean(xa, dim=2, keepdim=True)
        avg2 = torch.mean(xv_lau, dim=2, keepdim=True)
        avg3 = torch.mean(xv_egh, dim=2, keepdim=True)
        avg4 = torch.mean(xi, dim=2, keepdim=True)

        attn_input = torch.cat([avg1, avg2, avg3, avg4], dim=2)  # [B, T, 4]
        attn_logits = self.proj(attn_input)  # [B, T, 4]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, T, 4]

        aw1 = attn_weights[:, :, 0:1]
        aw2 = attn_weights[:, :, 1:2]
        aw3 = attn_weights[:, :, 2:3]
        aw4 = attn_weights[:, :, 3:4]

        x = torch.cat([aw1 * xa, aw2 * xv_lau, aw3 * xv_egh, aw4 * xi], dim=-1)
        x = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv2(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class AdaptiveAttentionMambaFusion(nn.Module):
    """
    This class implements the Adaptive Attention Mamba Fusion Module (AAMFM).
    This module cointains a Modal-wise Attention Block and a Multimodal Mamba Encoder (MME).
    """

    def __init__(
        self, mm_input_size=256, mm_output_sizes=[256], dropout=0.1, mamba_config=None
    ):
        super().__init__()

        self.MME = ResMambaBlock(
            mm_output_sizes, dropout=dropout, mamba_config=mamba_config
        )

        self.modal_attn = ModalWiseAttention(mm_input_size)

    def forward(self, xa, xv_lau, xv_egh, xi):
        x = self.modal_attn(xa, xv_lau, xv_egh, xi)
        x = self.MME(x)
        return x


class CafMamba(BaseNet):
    """
    This class implements the CAF-Mamba model for multimodal features.

    Args:
        audio_input_size (int): The input size of audio features, which is 128 for LMVD dataset.
        video_egh_input_size (int): The input size of eye-gaze-head features, which is 280+8+6 for LMVD dataset.
        video_lau_input_size (int): The input size of facial landmarks + Action Units features, which is 136+35 for LMVD dataset.
        mm_input_size (int): The input size of the multimodal features. Default is 256.
        mm_output_sizes (list): The output sizes of the multimodal features. Default is [256].
                                By default, only one ResMamba layer is used in each ResMamba block.
        dropout (float): The dropout rate. Default is 0.1.
        mamba_config (dict): The configuration of the Mamba layers.

    Returns:
         x (torch.Tensor): After self.classifier the predicted output of the model is obtained, which is transformed to binary depression labels in the main.py file.
    """

    def __init__(
        self,
        audio_input_size=128,
        video_egh_input_size=294,
        video_lau_input_size=171,
        mm_input_size=256,
        mm_output_sizes=[256],
        dropout=0.1,
        mamba_config=None,
    ):
        super().__init__()

        # transform input features to same dimension, defult 256 dims
        self.conv_audio = nn.Conv1d(
            audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False
        )
        self.conv_video_lau = nn.Conv1d(
            video_lau_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False
        )
        self.conv_video_egh = nn.Conv1d(
            video_egh_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False
        )

        self.UFM = UnimodalExtractionModule(
            mm_output_sizes, dropout=dropout, mamba_config=mamba_config
        )

        # Cross-Modal Interaction Mamba Encoder
        self.CIME = nn.ModuleList()
        self.num_CIME = 1
        for _ in range(self.num_CIME):
            self.CIME.append(
                ResMambaBlock(
                    mm_output_sizes, dropout=dropout, mamba_config=mamba_config
                )
            )

        self.attenvector = AdaptiveAttentionMambaFusion(
            mm_input_size, mm_output_sizes, dropout, mamba_config=mamba_config
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.drop = nn.Dropout(0.5)
        self.output = nn.Linear(mm_output_sizes[-1], 1)

        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video_lau.weight.data)
        nn.init.xavier_uniform_(self.conv_video_egh.weight.data)

    def feature_extractor(self, x, padding_mask=None):
        xa = x[:, :, 465:]  # 128 dims, audio features in LMVD
        xv_egh = x[:, :, :294]  # 294 dims, eye-gaze-head features in LMVD
        xv_lau = x[:, :, 294:465]  # 171 dims, facial landmarks + AUs in LMVD

        # transform input features to specific dimension
        xa = self.conv_audio(xa.permute(0, 2, 1)).permute(0, 2, 1)
        xv_egh = self.conv_video_egh(xv_egh.permute(0, 2, 1)).permute(0, 2, 1)
        xv_lau = self.conv_video_lau(xv_lau.permute(0, 2, 1)).permute(0, 2, 1)
        # initialize intermodal representation
        xi = torch.zeros_like(xa)

        xa, xv_lau, xv_egh = self.UFM(xa, xv_lau, xv_egh)

        for i in range(self.num_CIME):
            xi = self.CIME[i](xi + xa + xv_lau + xv_egh)

        x = self.attenvector(xa, xv_lau, xv_egh, xi)

        if padding_mask is not None:
            x = x * (padding_mask.unsqueeze(-1).float())
            x = x.sum(dim=1) / (padding_mask.unsqueeze(-1).float()).sum(
                dim=1, keepdim=False
            )

        else:
            x = self.pool(x.permute(0, 2, 1)).squeeze(-1)

        x = self.drop(x)
        return x

    def classifier(self, x):
        return self.output(x)
