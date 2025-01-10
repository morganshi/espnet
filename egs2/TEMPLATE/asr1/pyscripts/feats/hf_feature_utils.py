import json
import logging
import os
import re
import sys
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.dataset import ESPnetDataset
from espnet.utils.cli_writers import file_writer_helper
from transformers import AutoModelForCTC, AutoFeatureExtractor

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("hf_feature_loader")


def format_feature_conf_str(feature_conf: str):
    # 1. removing any extraneous white spaces
    feature_conf = re.sub(r"\s", "", feature_conf)
    # Surrounding any word/path with "
    feature_conf = re.sub(r"([\w\.\-/]+)", r'"\1"', feature_conf)
    # Replacing = with :
    feature_conf = re.sub(r"=", ": ", feature_conf)
    try:
        feature_conf = json.loads(feature_conf)
    except Exception as e:
        logger.warning(f"Failure in parsing feature_conf {feature_conf}")
        raise e
    return feature_conf


def build_data_iterator(
    rspecifier: str,
    in_filetype: str,
    utt2num_samples: str,
    batch_bins: Optional[int] = 1,
):
    dataset = ESPnetDataset(
        [(rspecifier[4:], "speech", in_filetype)],
        preprocess=None,
    )
    sampler = NumElementsBatchSampler(
        batch_bins=batch_bins,
        shape_files=[utt2num_samples],
    )
    batches = list(sampler)
    iterator = SequenceIterFactory(
        dataset=dataset,
        batches=batches,
        collate_fn=CommonCollateFn(float_pad_value=0.0, int_pad_value=-1),
        num_workers=2,
    ).build_iter(0)
    return iterator


def dump_feature(
    reader,
    in_filetype: str,
    rspecifier: str,
    out_filetype: str,
    wspecifier: str,
    utt2num_samples: Optional[str] = None,
    batch_bins: Optional[int] = None,
    write_num_frames: bool = None,
):
    assert os.path.exists(utt2num_samples), f"{utt2num_samples} does not exist."

    iterator = build_data_iterator(rspecifier, in_filetype, utt2num_samples, batch_bins)

    with file_writer_helper(
        wspecifier,
        filetype=out_filetype,
        write_num_frames=write_num_frames,
    ) as writer:
        for utt_ids, data in iterator:
            feats, feats_lens = reader.get_feats(data["speech"], data["speech_lengths"])
            for idx, utt in enumerate(utt_ids):
                writer[utt] = feats[idx][: feats_lens[idx]].numpy()
    logger.info("finished successfully")


class BaseFeatureReader(object):
    def __init__(self):
        raise NotImplementedError

    def load_audio(self, path: str, ref_len: Optional[int] = None):
        wav, sr = sf.read(path)
        # assert sr == self.sample_rate, sr
        if sr != self.sample_rate:
            logging.warning(
                "sampling rate mismatch between "
                "the requirements of feature extractor {} "
                "and source wav {},"
                "conduct resampling".format(self.sample_rate, sr)
            )
            wav = librosa.resample(wav, sr, self.sample_rate, scale=True)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def preprocess_data(
        self,
        data: Union[str, np.ndarray, list, torch.Tensor],
        data_lens: Union[int, List[int], torch.Tensor],
        ref_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return data, data_lens
        elif isinstance(data, str):
            batch_size = 1
            x = self.load_audio(data, ref_len=ref_len)
        elif isinstance(data, np.ndarray):
            batch_size = 1
            x = data
        else:
            raise TypeError(f"Unexpected data type of argument 1: {type(data)}.")
        x = torch.from_numpy(x).view(batch_size, -1).float()
        x_lens = torch.tensor([data_lens]).long()
        return x, x_lens

    def get_feats(
        self, data: torch.Tensor, data_lens: torch.Tensor, ref_len: Optional[int] = None
    ):
        raise NotImplementedError



class HuggingFaceFeatureReader(BaseFeatureReader):
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        audio_sample_rate: int = 16000,
        download_dir: str = None,
        multilayer_feature: bool = False,
        layer: int = -1,
        use_gpu: bool = True,
    ):  
        self.processor = AutoFeatureExtractor.from_pretrained('/data/mohan/workdir/download/wavlm-large')
        self.model = AutoModelForCTC.from_pretrained(download_dir)

        self.layer = layer

        self.sample_rate = fs
        self.audio_sample_rate = audio_sample_rate
        if self.sample_rate != self.audio_sample_rate:
            logging.warning("The audio sample rate is different from feat extractor")
            self.resample = torchaudio.transforms.Resample(
                orig_freq=audio_sample_rate, new_freq=fs
            )
        else:
            self.resample = None

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.conv_layers = [
            {"K": 10, "S": 5, "P": 3},
            {"K": 3, "S": 2, "P": 1},
            {"K": 3, "S": 2, "P": 1},
            {"K": 3, "S": 2, "P": 1},
            {"K": 3, "S": 2, "P": 1},
            {"K": 2, "S": 2, "P": 0},
            {"K": 2, "S": 2, "P": 0},
        ]

    def get_feats(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
        ref_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # import ipdb;ipdb.set_trace()
            input_values = []
            batch = data.shape[0]
            assert data_lens.shape[0] == batch

            for i in range(batch):
                input_values.append(data[i][:data_lens[i]].cpu().numpy())

            input_values = self.processor(input_values, return_tensors="pt", sampling_rate=self.sample_rate, padding=True).to(self.model.device)

            outputs = self.model.wavlm(
                **input_values,
                output_hidden_states=True,
            )
            
            feats = outputs[-1][self.layer]

            feats_lens = data_lens

            for i, conv_layer in enumerate(self.conv_layers):
                K, S, P = conv_layer["K"], conv_layer["S"], conv_layer["P"]
                feats_lens = (feats_lens + 2 * P - K) // S + 1


            gap = feats.shape[1] - torch.max(feats_lens)

            feats_lens = feats_lens + gap

            # x, x_lens = self.preprocess_data(data, data_lens)

            # if self.resample is not None:
            #     x = self.resample(x)
            #     x_lens = x_lens * self.sample_rate // self.audio_sample_rate

            # x = x.to(self.device)

            # outputs = self.model.wavlm(
            #     x,
            #     output_hidden_states=True,
            # )

            # feats = outputs[-1][self.layer] #shape: 1,80,1024
            # #len(outputs[-1])==25

            # feats_lens = torch.tensor([feats.shape[-2]]) # [80]

        feats = feats.cpu()
        feats_lens = feats_lens.cpu()
        return feats, feats_lens

if __name__=="__main__":
    reader_conf={
        "fs": 16000,
        "audio_sample_rate": 16000,
        "download_dir": "/data/mohan/workdir/espnet/egs2/myst/asr2/exp/wavlm-large-myst-fullfinetune/",
        "multilayer_feature": False,
        "layer": 24,
        "use_gpu": False
        }
    reader = HuggingFaceFeatureReader(**reader_conf)
    dump_feature(
        reader,
        in_filetype="sound",
        rspecifier="scp:/data/mohan/workdir/espnet/egs2/myst/asr2/dump/audio_raw/debug/wav.scp",
        out_filetype="mat",
        wspecifier="ark,scp:/data/mohan/workdir/espnet/egs2/myst/asr2/dump/audio_raw/debug/extract/feat.ark,/data/mohan/workdir/espnet/egs2/myst/asr2/dump/audio_raw/debug/extract/feat.scp",
        utt2num_samples="/data/mohan/workdir/espnet/egs2/myst/asr2/dump/audio_raw/debug/utt2num_samples",
        write_num_frames="ark,t:/data/mohan/workdir/espnet/egs2/myst/asr2/dump/audio_raw/debug/extract/utt2num_frames",
        batch_bins=1,
    )