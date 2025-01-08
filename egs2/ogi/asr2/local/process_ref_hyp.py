import sys
import os, json
if __name__=="__main__":
    src_hyp_file_path=sys.argv[1]
    src_ref_file_path=sys.argv[2]
    ori_hyp_file=open(src_hyp_file_path,'r')
    ori_hyp_lines=ori_hyp_file.readlines()
    ori_hyp_file.close()
    ori_ref_file=open(src_ref_file_path,'r')
    ori_ref_lines=ori_ref_file.readlines()
    ori_ref_file.close()

    out_path=os.path.dirname(src_hyp_file_path)
    
    os.system("mkdir -p "+out_path)
    single_word_hyp_file=open(out_path+'/text_hyp_single_word','w')
    single_word_ref_file=open(out_path+'/text_ref_single_word','w')
    multi_word_hyp_file=open(out_path+'/text_hyp_multi_word','w')
    multi_word_ref_file=open(out_path+'/text_ref_multi_word','w')

    n=len(ori_hyp_lines)
    assert n==len(ori_ref_lines)
    i=1
    
    for i in range(n):
        ori_hyp_array=ori_hyp_lines[i].strip().split(' ')
        ori_ref_array=ori_ref_lines[i].strip().split(' ')
        uttid=ori_hyp_array[0]
        assert uttid==ori_ref_array[0]

        if len(ori_hyp_array)>1:
            hyp=' '.join(ori_hyp_array[1:])
        else:
            hyp=' '

        if len(ori_ref_array)>1:
            ref=' '.join(ori_ref_array[1:])
        else:
            ref=' '

        hyp=hyp.lower().replace('$ ','').replace(' $','')
        ref=ref.lower().replace('$ ','').replace(' $','')

        if len(ref.split(' '))>1:
            single_word_hyp_file.write(uttid+' '+hyp+'\n')
            single_word_ref_file.write(uttid+' '+ref+'\n')
        else:
            multi_word_hyp_file.write(uttid+' '+hyp+'\n')
            multi_word_ref_file.write(uttid+' '+ref+'\n')

        i+=1
    single_word_hyp_file.close()
    single_word_ref_file.close()
    multi_word_hyp_file.close()
    multi_word_ref_file.close()