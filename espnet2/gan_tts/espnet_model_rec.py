# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based text-to-speech ESPnet model."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch < 1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetGANRECModel(AbsGANESPnetModel):
    """ESPnet model for GAN-based text-to-speech task."""

    @typechecked
    def __init__(
        self,
        tts: AbsGANTTS,
    ):
        """Initialize ESPnetGANTTSModel module."""
        super().__init__()
        self.tts = tts
        assert hasattr(
            tts, "generator"
        ), "generator module must be registered as tts.generator"
        assert hasattr(
            tts, "discriminator"
        ), "discriminator module must be registered as tts.discriminator"

    def forward(
        self,
        speech_token: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return generator or discriminator loss with dict format.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            forward_generator (bool): Whether to forward generator.
            kwargs: "utt_id" is among the input.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # Make batch for tts inputs
        batch = dict(
            speech_token=speech_token,
            speech_token_lengths=speech_token_lengths,
            forward_generator=forward_generator,
        )

        # Update batch for additional auxiliary inputs
        if self.tts.require_raw_speech:
            batch.update(speech=speech, speech_lengths=speech_lengths)
        return self.tts(**batch)


    def collect_feats(
            self,
            speech_token: torch.Tensor,
            speech_token_lengths: torch.Tensor,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            forward_generator: bool = True,
            **kwargs,
        ) -> Dict[str, torch.Tensor]:

            # store in dict
            feats_dict = {}

            return feats_dict
