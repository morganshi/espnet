from transformers import Wav2Vec2Processor, WavLMForCTC
import torch
import torchaudio
import soundfile as sf
from jiwer import wer
import os
from english_norm import EnglishTextNormalizer

model_name = "/data/mohan/workdir/espnet/egs2/librispeech_100/asr2/exp/wavlm-libri-clean-100h-large"
processor = Wav2Vec2Processor.from_pretrained("/data/mohan/workdir/espnet/egs2/librispeech_100/asr2/exp/wavlm-libri-clean-100h-large")
model = WavLMForCTC.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

wav_scp_path = "/data/mohan/workdir/espnet/egs2/librispeech_100/asr1/dump/raw/test_clean/wav.scp"
wav_dict = {}

with open(wav_scp_path, "r") as f:
    for line in f:
        utt_id, wav_path = line.strip().split(maxsplit=1)
        wav_dict[utt_id] = wav_path

transcriptions = {}

# normalize_file = os.path.join("/data/mohan/workdir/espnet/egs2/librispeech_100/asr2/exp/wavlm-libri-clean-100h-large/", "normalizer.json")
# with open(normalize_file, encoding="utf-8") as vocab_handle:
#     import json
#     english_spelling_normalizer = json.load(vocab_handle)
# normalizer = EnglishTextNormalizer(english_spelling_normalizer)

print("Starting inference...\n")

for i, (utt_id, wav_path) in enumerate(wav_dict.items(), start=1):
    speech_array, sampling_rate = sf.read(wav_path)
    # import ipdb;ipdb.set_trace()
    # 需要转换为 16kHz 采样率
    if sampling_rate != 16000:
        speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(torch.tensor(speech_array)).numpy()

    # 预处理音频
    input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)

    # 进行推理
    with torch.no_grad():
        logits = model(input_values).logits

    # 解码得到文本
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # transcription = normalizer(transcription)
    # 存储转录结果
    transcriptions[utt_id] = transcription

    # 实时打印解码信息
    print(f"[{i}/{len(wav_dict)}] Decoded {utt_id}: {transcription}")

# 保存转录结果到 hyp.trn
hyp_trn_path = "/data/mohan/workdir/espnet/egs2/librispeech_100/asr1/wavlm_hf_infer_results/test_clean/text_hyp"
with open(hyp_trn_path, "w") as f:
    for utt_id, transcription in transcriptions.items():
        f.write(f"{utt_id} {transcription}\n")

print(f"Transcriptions saved to {hyp_trn_path}")

# 读取参考文本
ref_texts = {}

ref_trn_path = "/data/mohan/workdir/espnet/egs2/librispeech_100/asr1/dump/raw/test_clean/text"
with open(ref_trn_path, "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            utt_id, ref_text = parts
            # ref_text = normalizer(ref_text)
            ref_texts[utt_id] = ref_text

# 计算 WER
hypotheses = []
references = []

for utt_id in ref_texts:
    if utt_id in transcriptions:
        hypotheses.append(transcriptions[utt_id].lower())  # 统一小写
        references.append(ref_texts[utt_id].lower())

wer_score = wer(references, hypotheses)
print(f"WER: {wer_score:.2%}")
