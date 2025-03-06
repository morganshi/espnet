import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend

from transformers import AutoModelForCTC, AutoFeatureExtractor

import numpy

import math

from vector_quantize_pytorch import FSQ

import joblib

import torch.nn.utils.rnn as rnn_utils

from transformers import WavLMModel

from transformers.modeling_outputs import BaseModelOutput


class WavLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)
        position_bias = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_bias,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_bias=position_bias,
                        output_attentions=output_attentions,
                        index=i,
                    )

                hidden_states, position_bias = layer_outputs[:2]

            if skip_the_layer:
                layer_outputs = (None, None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class CustomWavLM(torch.nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus", fusion_fn=None):
        super(CustomWavLM, self).__init__()
        # 加载预训练的 WavLM
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.encoder_layers = self.wavlm.encoder.layers  # 提取所有层
        self.embedding_layer = self.wavlm.feature_extractor  # 提取输入嵌入层
        self.fusion_fn = fusion_fn  # 融合函数，用于融合其他特征
        self.num_layers = len(self.encoder_layers)
        self.layer_norm = self.wavlm.encoder.layer_norm

    def forward(self, input_values, attention_mask=None, additional_features=None):
        # 获取初始嵌入
        hidden_states = self.embedding_layer(input_values)
        
        if attention_mask is not None:
            attention_mask = self.wavlm._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )

        # 对嵌入应用每层 Transformer
        all_layer_outputs = []
        for i, layer in enumerate(self.encoder_layers):
            # 自定义逻辑：在每层中融合其他特征
            if self.fusion_fn and additional_features is not None:
                hidden_states = self.fusion_fn(hidden_states, additional_features, layer_idx=i)
            
            # Transformer 层的前向传播
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            all_layer_outputs.append(hidden_states)
        
        # 最后应用层归一化
        hidden_states = self.layer_norm(hidden_states)
        all_layer_outputs.append(hidden_states)
        
        return all_layer_outputs


class HfCTCKMATTFSQFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        download_dir: Optional[str] = None,
        km_path: Optional[str] = None,
        layer: int = -1,
    ):
        super().__init__()

        self.upstream = AutoModelForCTC.from_pretrained(download_dir)
        self.processor = AutoFeatureExtractor.from_pretrained('/data/mohan/workdir/download/wavlm-large')

        self.upstream.freeze_feature_encoder()

        self.quantizer = FSQ(levels = [5,5,5,4,4], dim = self.upstream.config.hidden_size)

        self.layer = layer

        if isinstance(fs, str):
            self.sample_rate = humanfriendly.parse_size(fs)
        else:
            self.sample_rate = fs

        if self.sample_rate != 16000:
            logging.warning(
                "All the upstream models in HF now only support 16 kHz audio."
            )

        self.km_model = joblib.load(km_path)

        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=self.upstream.config.hidden_size, num_heads=8, batch_first=True)

        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
        self.frontend_type = "hf_ctc"

        self.conv_layers = [
            {"K": 10, "S": 5, "P": 3},
            {"K": 3, "S": 2, "P": 1},
            {"K": 3, "S": 2, "P": 1},
            {"K": 3, "S": 2, "P": 1},
            {"K": 3, "S": 2, "P": 1},
            {"K": 2, "S": 2, "P": 0},
            {"K": 2, "S": 2, "P": 0},
        ]

    def _tile_representations(self, feature):
        """Tile up the representations by `tile_factor`.

        Input - sequence of representations
                shape: (batch_size, seq_len, feature_dim)

        Output - sequence of tiled representations
                 shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(feature.shape) == 3
        ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature

    def output_size(self) -> int:
        return self.upstream.config.hidden_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor, input_km_id: torch.Tensor, input_km_id_lengths: torch.Tensor, extract_token = False
    ):
        input_values = []
        batch = input.shape[0]
        assert input_lengths.shape[0] == batch

        for i in range(batch):
            input_values.append(input[i][:input_lengths[i]].cpu().numpy())

        input_values = self.processor(input_values, return_tensors="pt", sampling_rate=self.sample_rate, padding=True).to(self.upstream.device)
        
        outputs = self.upstream.wavlm(
            **input_values,
            output_hidden_states=True,
        )

        feats = outputs[-1][self.layer]

        feats_lens = input_lengths

        # for stride in self.upstream.config.conv_stride:
        #     feats_lens = feats_lens // stride + 1

        for i, conv_layer in enumerate(self.conv_layers):
            K, S, P = conv_layer["K"], conv_layer["S"], conv_layer["P"]
            feats_lens = (feats_lens + 2 * P - K) // S + 1


        gap = feats.shape[1] - torch.max(feats_lens)

        feats_lens = feats_lens + gap

        if isinstance(input_km_id, torch.Tensor):
            input_km_id = input_km_id.cpu().numpy()  # 转为 NumPy 数组
        
        km_feats = self.km_model.cluster_centers_[input_km_id]

        km_feats = torch.from_numpy(km_feats).to(self.upstream.device)

        max_len = km_feats.shape[1]

        padding_mask = torch.arange(max_len).expand(batch, max_len).to(self.upstream.device) >= input_km_id_lengths.unsqueeze(1)

        feats_att, _ = self.cross_attn(feats, km_feats, km_feats, key_padding_mask=padding_mask)

        feats = feats + feats_att
        
        feats_codes, indices = self.quantizer(feats)

        if extract_token:
            return indices
        else:
            return feats_codes, feats_lens
