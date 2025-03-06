import sys,os
if __name__=="__main__":
    path=sys.argv[1]
    wav_scp_file=open(path+"/wav.scp",'r')
    wav_scp=wav_scp_file.readlines()
    wav_scp_file.close()
    
    utt2spk_file=open(path+"/utt2spk",'w')

    for line in wav_scp:
        uttid=line.strip().split(' ')[0]
        spk=uttid[:5]
        utt2spk_file.write(uttid+" "+spk+"\n")

    utt2spk_file.close()