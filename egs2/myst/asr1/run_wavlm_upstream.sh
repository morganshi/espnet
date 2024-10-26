#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/train_asr_wavlm_ebranchformer_noctc.yaml
inference_config=conf/decode_asr_noctc.yaml


CUDA_VISIBLE_DEVICES="0,1"    \
./asr.sh \
    --stage 13   \
    --stop_stage 13  \
    --lang en \
    --ngpu 2 \
    --nj 16 \
    --gpu_inference false \
    --inference_nj 8 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn \
    --audio_format "flac" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
