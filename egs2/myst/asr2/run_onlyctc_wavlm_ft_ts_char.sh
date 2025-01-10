#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

CUDA_VISIBLE_DEVICES="0"

kmeans_feature="wavlm_large_finetune/24"  # use model_type/layer_index
nclusters=2000

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en

train_set="train"
train_dev="dev"
test_sets="dev test"

# test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_discrete_asr_e_branchformer1_onlyctc_1gpu_lr5e-4.yaml
inference_config=conf/decode_ctc1.0_greedy.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"

CUDA_VISIBLE_DEVICES="2"    \
./asr2_hf.sh \
    --stage 14   \
    --stop_stage 14  \
    --gpu_kmeans true  \
    --portion 1.0   \
    --kmeans_opts "--batch_bins 1 --nj 8" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --ngpu 1 \
    --nj 4  \
    --inference_nj 4    \
    --inference_asr_model "valid.cer_ctc.best.pth"   \
    --gpu_inference true    \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "char" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "dump/raw/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "dump/raw/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "dump/raw/${train_set}/text.${tgt_case}.${tgt_lang}" "$@"
