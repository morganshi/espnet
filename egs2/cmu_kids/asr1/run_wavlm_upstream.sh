#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail


train_set="train"
valid_set="dev"
test_sets="dev test"

encoder=transformer
frontend=wavlm
asr_config=conf/train_asr_wavlm_ebranchformer_onlyctc.yaml
inference_config=conf/decode_asr_ctc_greedy.yaml

nbpe=5000
bpemode=unigram

# if your sox supports flac file, set local_data_opts and audio_format as below.
#local_data_opts=""
#audio_format=flac

# if your sox does not support flac file, set local_data_opts and audio_format as below.
local_data_opts="--sph2wav true"
audio_format=wav

# set a higher min_wav_duration to avoid TooShortUttError in stage 11
min_wav_duration=0.3

CUDA_VISIBLE_DEVICES="0"    \
./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 4 \
    --inference_nj 1 \
    --gpu_inference true \
    --token_type bpe \
    --stage 13 \
    --stop_stage 13 \
    --bpemode "${bpemode}" \
    --nbpe "${nbpe}" \
    --max_wav_duration 30 \
    --use_lm false \
    --feats_normalize utterance_mvn \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.cer_ctc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --local_data_opts "${local_data_opts}" \
    --audio_format ${audio_format} \
    --min_wav_duration ${min_wav_duration} \
    "$@"
