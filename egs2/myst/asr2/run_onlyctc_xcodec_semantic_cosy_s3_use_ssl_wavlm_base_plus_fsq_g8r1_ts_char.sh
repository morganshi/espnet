#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

CUDA_VISIBLE_DEVICES="0"

kmeans_feature="wavlm_large_finetune/24"  # use model_type/layer_index
nclusters=2000

src_lang=xcodec_semantic_cosy_s3_use_ssl_wavlm_base_plus_fsq_g8r1
tgt_lang=en

train_set="train"
train_dev="dev"
test_sets="test"

# test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_discrete_asr_codec_concat_init_freeze_e_branchformer1_onlyctc_1gpu_lr5e-4.yaml
inference_config=conf/decode_ctc1.0_greedy.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"

# 定义目录
base_dir="dump/codec_infer_data"
dir="${base_dir}/xcodec_semantic_cosy_s3_use_ssl_wavlm_base_plus_fsq_g8r1"

# 生成路径数组
src_codebook_paths=()
for i in {0..7}; do
    src_codebook_paths+=("${dir}/codebook_${i}.npy")
done

# 转换为空格分隔的字符串（保持原始格式）
src_codebook_paths="${src_codebook_paths[*]}"

# src_codebook_paths="codec_infer_data/xcodec_wavlm_base_plus_fsq_g8r1/codebook_0.npy codec_infer_data/xcodec_wavlm_base_plus_fsq_g8r1/codebook_1.npy codec_infer_data/xcodec_wavlm_base_plus_fsq_g8r1/codebook_2.npy codec_infer_data/xcodec_wavlm_base_plus_fsq_g8r1/codebook_3.npy codec_infer_data/xcodec_fsq_g8r1/codebook_4.npy codec_infer_data/xcodec_wavlm_base_plus_fsq_g8r1/codebook_5.npy codec_infer_data/xcodec_wavlm_base_plus_fsq_g8r1/codebook_6.npy codec_infer_data/xcodec_wavlm_base_plus_fsq_g8r1/codebook_7.npy"

CUDA_VISIBLE_DEVICES="0"    \
./asr2_codec.sh \
    --stage 15   \
    --stop_stage 15  \
    --src_token_size 1000   \
    --src_codebook_size 8   \
    --train_speech_token_scp ${dir}/train/speech_token.scp    \
    --valid_speech_token_scp ${dir}/dev/speech_token.scp    \
    --test_speech_token_scp ${dir}/test/speech_token.scp    \
    --train_text_scp /data/mohan/workdir/espnet/egs2/myst/asr2/dump/raw/train/text.ts.en    \
    --valid_text_scp /data/mohan/workdir/espnet/egs2/myst/asr2/dump/raw/dev/text.ts.en    \
    --test_text_scp /data/mohan/workdir/espnet/egs2/myst/asr2/dump/raw/test/text.ts.en    \
    --src_codebook_paths "${src_codebook_paths}"  \
    --asr_stats_dir /data/mohan/workdir/espnet/egs2/myst/asr2/exp/asr_stats_raw_ts_xcodec_fsq_g8r1_char_bpe5000 \
    --ngpu 1 \
    --nj 8  \
    --inference_nj 1    \
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
