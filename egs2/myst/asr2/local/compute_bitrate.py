import numpy as np
if __name__=="__main__":
    speech_shape_file=open("/data/mohan/workdir/espnet/egs2/myst/asr1/exp/asr_stats_raw_en_bpe5000/train/speech_shape",'r')
    speech_shape_scp=speech_shape_file.readlines()
    speech_shape_file.close()
    token_shape_file=open("/data/mohan/workdir/espnet/egs2/myst/asr2/exp/asr_stats_raw_rm_wavlm_large_21_km2000_bpe6000_bpe5000/train/src_text_shape",'r')
    token_shape_scp=token_shape_file.readlines()
    token_shape_file.close()
    total_token_len=0
    total_frame_len=0
    sample_rate=16000
    for line in speech_shape_scp:
        total_frame_len+=int(line.strip().split(' ')[-1])
    
    for line in token_shape_scp:
        total_token_len+=int(line.strip().split(' ')[-1])

    bitrate = float(total_token_len*np.log2(2000)) / (float(total_frame_len)/float(sample_rate))
    print(bitrate)