from typing import Tuple, List, Union

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
import numpy as np


class CodebookEmbedding(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    @typechecked
    def __init__(
        self,
        init_code=False,
        input_size: int = 400,
        embed_dim: int = 400,
        num_codebooks: int = 8,
        concat: bool = False,
        npy_paths: Union[Tuple[str, ...], List[str]] = [],
        freeze_codebook: bool = False,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """
        super().__init__()
        self.codebooks = torch.nn.ModuleList()
        self.concat = concat

        if init_code:
            for npy_path in npy_paths:
                codebook_array = np.load(npy_path)
                codebook_tensor = torch.from_numpy(codebook_array).float()
                embed_dim = codebook_tensor.shape[1]
                
                # 仅创建Embedding层
                embed = torch.nn.Embedding.from_pretrained(
                    codebook_tensor,
                    freeze=freeze_codebook
                )
                self.codebooks.append(embed)
            # 设置concat后的维度
            self.embed_dim = embed_dim * len(npy_paths) if concat else embed_dim
        else:
            self.input_size = input_size
            self.embed_dim = embed_dim
            if concat:
                self.embed_dim = embed_dim * num_codebooks
            for _ in range(num_codebooks):
                embed = torch.nn.Embedding(input_size, embed_dim)
                self.codebooks.append(embed)
        
        # 添加统一的位置编码层
        self.pos_enc = pos_enc_class(self.embed_dim, positional_dropout_rate)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            input: Input (B, T) or (B, T,D), with D.
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T, D).
            Tensor: Output lengths within batch.
        """
        # 统一输入格式处理
        if input.dim() == 2:
            input = input.unsqueeze(0)  # (1, B, T)
        C, B, T = input.shape
        assert C == len(self.codebooks), f"Input has {C} codebooks, but model has {len(self.codebooks)}"

        embeddings = []
        for i in range(C):
            # 各codebook单独处理
            embed = self.codebooks[i](input[i])  # (B, T, D)
            embeddings.append(embed)
        
        # 合并操作
        if self.concat:
            merged = torch.cat(embeddings, dim=-1)  # (B, T, D*C)
        else:
            merged = torch.stack(embeddings, dim=0).mean(dim=0)  # (B, T, D)
        
        # 应用统一的位置编码
        merged = self.pos_enc(merged)
        return merged, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim