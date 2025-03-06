#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/train_asr_wavlm_ogi_fsq_ebranchformer_onlyctc_lr1e-4.yaml
inference_config=conf/decode_asr_ctc_greedy.yaml


CUDA_VISIBLE_DEVICES="0,1,2,3"    \
./asr_fsq.sh \
    --stage 13   \
    --stop_stage 13  \
    --lang en \
    --ngpu 4 \
    --nj 4 \
    --gpu_inference false \
    --inference_asr_model "valid.cer_ctc.best.pth"  \
    --inference_nj 4 \
    --gpu_inference true    \
    --nbpe 200 \
    --max_wav_duration 30 \
    --feats_normalize None \
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
