# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Joint text-to-wav module for end-to-end training."""

from typing import Any, Dict

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.hifigan import (
    HiFiGANGenerator,
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
    HiFiGANPeriodDiscriminator,
    HiFiGANScaleDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
)
from espnet2.gan_tts.melgan import MelGANGenerator, MelGANMultiScaleDiscriminator
from espnet2.gan_tts.melgan.pqmf import PQMF
from espnet2.gan_tts.parallel_wavegan import (
    ParallelWaveGANDiscriminator,
    ParallelWaveGANGenerator,
)
from espnet2.gan_tts.style_melgan import StyleMelGANDiscriminator, StyleMelGANGenerator
from espnet2.gan_tts.utils import get_random_segments, get_segments
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer
from espnet2.mt.frontend.embedding import Embedding

AVAILABLE_TEXT2MEL = {
    "tacotron2": Tacotron2,
    "transformer": Transformer,
    "fastspeech": FastSpeech,
    "fastspeech2": FastSpeech2,
}
AVAILABLE_VOCODER = {
    "hifigan_generator": HiFiGANGenerator,
    "melgan_generator": MelGANGenerator,
    "parallel_wavegan_generator": ParallelWaveGANGenerator,
    "style_melgan_generator": StyleMelGANGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
    "melgan_multi_scale_discriminator": MelGANMultiScaleDiscriminator,
    "parallel_wavegan_discriminator": ParallelWaveGANDiscriminator,
    "style_melgan_discriminator": StyleMelGANDiscriminator,
}


class WeightedCodebookEmbedding(nn.Module):
    def __init__(self, nq, token_size, embed_dim):
        super().__init__()
        self.nq = nq
        self.embed_dim = embed_dim
        # 定义 nq 个独立的 embedding 层
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=token_size, embedding_dim=embed_dim) 
            for _ in range(nq)
        ])
        # 可训练权重，初始化均匀分布
        self.weights = nn.Parameter(torch.ones(nq) / nq)

    def forward(self, tokens):
        """
        tokens: Tensor of shape (batch_size, nq * token_length)
        自动 reshape 成 (batch_size, nq, token_length)
        """
        batch_size, total_length = tokens.shape
        assert total_length % self.nq == 0, f"Input token length {total_length} must be divisible by nq={self.nq}"
        
        token_length = total_length // self.nq  # 计算 token_length

        # 重新 reshape: (batch_size, nq, token_length)
        tokens = tokens.view(batch_size, self.nq, token_length)

        if self.nq == 1:
            return self.embedding_layers[0](tokens.squeeze(1))  # (batch_size, token_length, embed_dim)

        # 获取所有 codebook 的 embedding
        embeddings = torch.stack(
            [self.embedding_layers[i](tokens[:, i, :]) for i in range(self.nq)], 
            dim=0
        )  # embeddings 形状: (nq, batch_size, token_length, embed_dim)

        # 计算加权和
        weighted_sum = torch.einsum("i,ibtd->btd", self.weights, embeddings)  # (batch_size, token_length, embed_dim)
        return weighted_sum



class UNIT2Wav(AbsGANTTS):
    """General class to jointly train text2mel and vocoder parts."""

    @typechecked
    def __init__(
        self,
        nq: int,
        token_size: int,
        segment_size: int = 32,
        sampling_rate: int = 22050,
        vocoder_type: str = "hifigan_generator",
        vocoder_params: Dict[str, Any] = {
            "in_channels": 80,
            "out_channels": 1,
            "channels": 512,
            "global_channels": -1,
            "kernel_size": 7,
            "upsample_scales": [8, 8, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "use_additional_convs": True,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
        },
        use_pqmf: bool = False,
        pqmf_params: Dict[str, Any] = {
            "subbands": 4,
            "taps": 62,
            "cutoff_ratio": 0.142,
            "beta": 9.0,
        },
        # discriminator related
        discriminator_type: str = "hifigan_multi_scale_multi_period_discriminator",
        discriminator_params: Dict[str, Any] = {
            "scales": 1,
            "scale_downsample_pooling": "AvgPool1d",
            "scale_downsample_pooling_params": {
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
            },
            "scale_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [15, 41, 5, 3],
                "channels": 128,
                "max_downsample_channels": 1024,
                "max_groups": 16,
                "bias": True,
                "downsample_scales": [2, 2, 4, 4, 1],
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            "follow_official_norm": False,
            "periods": [2, 3, 5, 7, 11],
            "period_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
        },
        # loss related
        generator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        discriminator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        use_feat_match_loss: bool = True,
        feat_match_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        use_mel_loss: bool = True,
        mel_loss_params: Dict[str, Any] = {
            "fs": 22050,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": None,
            "window": "hann",
            "n_mels": 80,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
    ):
        """Initialize JointText2Wav module.

        Args:
            idim (int): Input vocabrary size.
            odim (int): Acoustic feature dimension. The actual output channels will
                be 1 since the model is the end-to-end text-to-wave model but for the
                compatibility odim is used to indicate the acoustic feature dimension.
            segment_size (int): Segment size for random windowed inputs.
            sampling_rate (int): Sampling rate, not used for the training but it will
                be referred in saving waveform during the inference.
            text2mel_type (str): The text2mel model type.
            text2mel_params (Dict[str, Any]): Parameter dict for text2mel model.
            use_pqmf (bool): Whether to use PQMF for multi-band vocoder.
            pqmf_params (Dict[str, Any]): Parameter dict for PQMF module.
            vocoder_type (str): The vocoder model type.
            vocoder_params (Dict[str, Any]): Parameter dict for vocoder model.
            discriminator_type (str): Discriminator type.
            discriminator_params (Dict[str, Any]): Parameter dict for discriminator.
            generator_adv_loss_params (Dict[str, Any]): Parameter dict for generator
                adversarial loss.
            discriminator_adv_loss_params (Dict[str, Any]): Parameter dict for
                discriminator adversarial loss.
            use_feat_match_loss (bool): Whether to use feat match loss.
            feat_match_loss_params (Dict[str, Any]): Parameter dict for feat match loss.
            use_mel_loss (bool): Whether to use mel loss.
            mel_loss_params (Dict[str, Any]): Parameter dict for mel loss.
            lambda_text2mel (float): Loss scaling coefficient for text2mel model loss.
            lambda_adv (float): Loss scaling coefficient for adversarial loss.
            lambda_feat_match (float): Loss scaling coefficient for feat match loss.
            lambda_mel (float): Loss scaling coefficient for mel loss.
            cache_generator_outputs (bool): Whether to cache generator outputs.

        """
        super().__init__()
        self.segment_size = segment_size
        self.use_pqmf = use_pqmf

        # define modules
        self.generator = torch.nn.ModuleDict()
        vocoder_class = AVAILABLE_VOCODER[vocoder_type]
        # if vocoder_type in ["hifigan_generator", "melgan_generator"]:
        #     vocoder_params.update(in_channels=vocoder_params["channels"])
        # elif vocoder_type in ["parallel_wavegan_generator", "style_melgan_generator"]:
        #     vocoder_params.update(aux_channels=vocoder_params["channels"])
        self.generator["vocoder"] = vocoder_class(
            **vocoder_params,
        )
        if self.use_pqmf:
            self.pqmf = PQMF(**pqmf_params)
        discriminator_class = AVAILABLE_DISCRIMINATORS[discriminator_type]
        self.discriminator = discriminator_class(
            **discriminator_params,
        )
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **generator_adv_loss_params,
        )
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params,
        )
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(
                **feat_match_loss_params,
            )
        self.use_mel_loss = use_mel_loss
        if self.use_mel_loss:
            self.mel_loss = MelSpectrogramLoss(
                **mel_loss_params,
            )

        # coefficients
        self.lambda_adv = lambda_adv
        if self.use_feat_match_loss:
            self.lambda_feat_match = lambda_feat_match
        if self.use_mel_loss:
            self.lambda_mel = lambda_mel

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate

        self.embedding_layer = WeightedCodebookEmbedding(nq=nq, token_size=token_size, embed_dim=vocoder_params["in_channels"])

    @property
    def require_raw_speech(self):
        """Return whether or not speech is required."""
        return True

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return False

    def forward(
        self,
        speech_token: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        if forward_generator:
            return self._forward_generator(
                speech_token=speech_token,
                speech_token_lengths=speech_token_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                **kwargs,
            )
        else:
            return self._forward_discrminator(
                speech_token=speech_token,
                speech_token_lengths=speech_token_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                **kwargs,
            )

    def _forward_generator(
        self,
        speech_token: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = speech.size(0)
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False

            stats = dict()
            feats_gen = self.embedding_layer(speech_token)
            # get random segments
            feats_gen_, start_idxs = get_random_segments(
                x=feats_gen.transpose(1, 2),
                x_lengths=speech_token_lengths,
                segment_size=self.segment_size,
            )
            # calculate vocoder outputs
            # import ipdb;ipdb.set_trace()
            speech_hat_ = self.generator["vocoder"](feats_gen_)
            if self.use_pqmf:
                speech_hat_ = self.pqmf.synthesis(speech_hat_)
        else:
            stats, speech_hat_, start_idxs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (stats, speech_hat_, start_idxs)

        speech_ = get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator["vocoder"].upsample_factor,
            segment_size=self.segment_size * self.generator["vocoder"].upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(speech_)

        # calculate losses
        adv_loss = self.generator_adv_loss(p_hat)
        adv_loss = adv_loss * self.lambda_adv
        loss = adv_loss
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p)
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats.update(feat_match_loss=feat_match_loss.item())
        if self.use_mel_loss:
            mel_loss = self.mel_loss(speech_hat_, speech_)
            mel_loss = self.lambda_mel * mel_loss
            loss = loss + mel_loss
            stats.update(mel_loss=mel_loss.item())

        stats.update(
            adv_loss=adv_loss.item(),
            loss=loss.item(),
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
        }

    def _forward_discrminator(
        self,
        speech_token: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform discriminator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = speech.size(0)
        speech = speech.unsqueeze(1)
        
        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False

            stats = dict()
            feats_gen = self.embedding_layer(speech_token)
            # get random segments
            feats_gen_, start_idxs = get_random_segments(
                x=feats_gen.transpose(1, 2),
                x_lengths=speech_token_lengths,
                segment_size=self.segment_size,
            )
            # calculate vocoder outputs
            speech_hat_ = self.generator["vocoder"](feats_gen_)
            if self.use_pqmf:
                speech_hat_ = self.pqmf.synthesis(speech_hat_)
        else:
            _, speech_hat_, start_idxs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (stats, speech_hat_, start_idxs)

        # parse outputs
        speech_ = get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator["vocoder"].upsample_factor,
            segment_size=self.segment_size * self.generator["vocoder"].upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_.detach())
        p = self.discriminator(speech_)

        # calculate losses
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            real_loss=real_loss.item(),
            fake_loss=fake_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
        }

    def inference(
        self,
        speech_token: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * feat_gan (Tensor): Generated feature tensor (T_text, C).

        """
        output_dict = dict()
        feats_gen = self.embedding_layer(speech_token)
        wav = self.generator["vocoder"].inference(feats_gen.squeeze(0))
        wav = wav.squeeze(1)
        if self.use_pqmf:
            wav = self.pqmf.synthesis(wav.unsqueeze(0).transpose(1, 2))
            wav = wav.squeeze(0).transpose(0, 1)
        output_dict.update(wav=wav)

        return output_dict
