src_path=$1
python local/process_ref_hyp.py  $src_path/dev/text /data/mohan/workdir/espnet/egs2/ogi/asr2/dump/raw/dev/text.ts.en
python local/compute_wer.py $src_path/dev/text_ref_single_word $src_path/dev/text_hyp_single_word $src_path/dev/text_single_word.cer
tail -n 3 $src_path/dev/text_single_word.cer > $src_path/dev/text_single_word.cer.txt
cat $src_path/dev/text_single_word.cer.txt
python local/compute_wer.py $src_path/dev/text_ref_multi_word $src_path/dev/text_hyp_multi_word $src_path/dev/text_multi_word.cer
tail -n 3 $src_path/dev/text_multi_word.cer > $src_path/dev/text_multi_word.cer.txt
cat $src_path/dev/text_multi_word.cer.txt

python local/process_ref_hyp.py  $src_path/test/text /data/mohan/workdir/espnet/egs2/ogi/asr2/dump/raw/test/text.ts.en
python local/compute_wer.py $src_path/test/text_ref_single_word $src_path/test/text_hyp_single_word $src_path/test/text_single_word.cer
tail -n 3 $src_path/test/text_single_word.cer > $src_path/test/text_single_word.cer.txt
cat $src_path/test/text_single_word.cer.txt
python local/compute_wer.py $src_path/test/text_ref_multi_word $src_path/test/text_hyp_multi_word $src_path/test/text_multi_word.cer
tail -n 3 $src_path/test/text_multi_word.cer > $src_path/test/text_multi_word.cer.txt
cat $src_path/test/text_multi_word.cer.txt