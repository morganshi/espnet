#!/usr/bin/env bash
# This model is trained only on the scripted speech subset.
# TODO: Incorporate spontaneous speech into the training.
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

CUDA_VISIBLE_DEVICES="0"    \
./asr.sh \
    --stage 10   \
    --stop_stage 10  \
    --lang en \
    --ngpu 1 \
    --nj 4 \
    --gpu_inference true \
    --inference_asr_model "valid.cer_ctc.best.pth"  \
    --inference_nj 4 \
    --nbpe 200 \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "dump/raw/${train_set}/text" \
    --bpe_train_text "dump/raw/${train_set}/text" "$@"
