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

class HfCTCFSQFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        download_dir: Optional[str] = None,
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
        self, input: torch.Tensor, input_lengths: torch.Tensor, extract_token = False
    ):
        # import ipdb;ipdb.set_trace()
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

        feats_codes, indices = self.quantizer(feats)

        if extract_token:
            return indices
        else:
            return feats_codes, feats_lens
