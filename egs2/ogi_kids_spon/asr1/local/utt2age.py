import sys,os
if __name__=="__main__":
    path=sys.argv[1]
    wav_scp_file=open(path+"/wav.scp",'r')
    wav_scp=wav_scp_file.readlines()
    wav_scp_file.close()
    ori_utt2age_file=open("/data/mohan/workdir/espnet/egs2/ogi_kids_spon/asr1/data/spont_all/utt2age",'r')
    ori_utt2age_scp=ori_utt2age_file.readlines()
    ori_utt2age_file.close()
    
    utt2age_dict={}
    for line in ori_utt2age_scp:
        uttid,age=line.strip().split(' ')
        utt2age_dict[uttid]=age

    utt2age_file=open(path+"/utt2age",'w')

    for line in wav_scp:
        uttid=line.strip().split(' ')[0]
        uttid_global=uttid.split('_')[0]
        age=utt2age_dict[uttid_global]
        utt2age_file.write(uttid+" "+age+"\n")

    utt2age_file.close()