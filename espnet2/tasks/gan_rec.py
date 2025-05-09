# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based text-to-speech task."""

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.espnet_model_rec import ESPnetGANRECModel
from espnet2.gan_tts.jets import JETS
from espnet2.gan_tts.joint import UNIT2Wav
from espnet2.gan_tts.vits import VITS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.gan_trainer import GANTrainer
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none


tts_choices = ClassChoices(
    "tts",
    classes=dict(
        vits=VITS,
        unit2wav=UNIT2Wav,
        jets=JETS,
    ),
    type_check=AbsGANTTS,
    default="vits",
)

class GANRECTask(AbsTask):
    """GAN-based text-to-speech task."""

    # GAN requires two optimizers
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --tts and --tts_conf
        tts_choices,
    ]

    # Use GANTrainer instead of Trainer
    trainer = GANTrainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetGANRECModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=0,
            not_sequence=["spembs", "sids", "lids"],
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_token", "speech")
        else:
            # Inference mode
            retval = ("speech_token",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = (
                "spembs",
                "durations",
                "pitch",
                "energy",
                "sids",
                "lids",
            )
        else:
            # Inference mode
            retval = (
                "spembs",
                "speech",
                "durations",
                "pitch",
                "energy",
                "sids",
                "lids",
            )
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetGANRECModel:
        # 3. TTS
        tts_class = tts_choices.get_class(args.tts)
        tts = tts_class(**args.tts_conf)

        # 5. Build model
        model = ESPnetGANRECModel(
            tts=tts,
            **args.model_conf,
        )
        return model

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: ESPnetGANRECModel,
    ) -> List[torch.optim.Optimizer]:
        # check
        assert hasattr(model.tts, "generator")
        assert hasattr(model.tts, "discriminator")

        # define generator optimizer
        optim_g_class = optim_classes.get(args.optim)
        if optim_g_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_g = fairscale.optim.oss.OSS(
                params=model.tts.generator.parameters(),
                optim=optim_g_class,
                **args.optim_conf,
            )
        else:
            optim_g = optim_g_class(
                model.tts.generator.parameters(),
                **args.optim_conf,
            )
        optimizers = [optim_g]

        # define discriminator optimizer
        optim_d_class = optim_classes.get(args.optim2)
        if optim_d_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_d = fairscale.optim.oss.OSS(
                params=model.tts.discriminator.parameters(),
                optim=optim_d_class,
                **args.optim2_conf,
            )
        else:
            optim_d = optim_d_class(
                model.tts.discriminator.parameters(),
                **args.optim2_conf,
            )
        optimizers += [optim_d]

        return optimizers
