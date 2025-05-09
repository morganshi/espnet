#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

opts=

train_set=train
valid_set=dev
test_sets="test"

train_config=conf/tuning/train_unit_hifigan.yaml
# train_config=conf/tuning/train_conformer_fastspeech2.yaml
inference_config=conf/tuning/decode_fastspeech.yaml

CUDA_VISIBLE_DEVICES=0
./rec2.sh \
    --stage 8   \
    --stop_stage 8  \
    --nj 8 \
    --inference_nj 1    \
    --tts_task gan_rec  \
    --feats_type raw \
    --tts_exp /data/mohan/workdir/espnet/egs2/myst/rec2/exp/km_2000_unit_hifigan \
    --train_token_path /data/mohan/workdir/espnet/egs2/myst/asr2/dump/extracted/wavlm_large_finetune/layer24/train/pseudo_labels_km2000.txt \
    --valid_token_path /data/mohan/workdir/espnet/egs2/myst/asr2/dump/extracted/wavlm_large_finetune/layer24/dev/pseudo_labels_km2000.txt \
    --test_token_path /data/mohan/workdir/espnet/egs2/myst/asr2/dump/extracted/wavlm_large_finetune/layer24/test/pseudo_labels_km2000.txt \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --inference_model train.mel_loss.best.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    ${opts} "$@"
