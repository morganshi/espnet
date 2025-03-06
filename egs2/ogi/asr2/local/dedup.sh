for dset in "train" "dev" "test"; do
    awk '
        (FILENAME==ARGV[1]) {
            out="";
            for (i=2; i<=NF; i++) {
                if ($i != $(i-1)) {
                    out = out" "$i;
                }
            }
            print($1, out);
        }' "/data/mohan/workdir/espnet/egs2/ogi/asr2/dump/extracted/wavlm_large_finetune/layer24/${dset}/pseudo_labels_km2000.txt" \
        > "/data/mohan/workdir/espnet/egs2/ogi/asr2/dump/extracted/wavlm_large_finetune/layer24/${dset}/pseudo_labels_km2000_rm.txt"
done
