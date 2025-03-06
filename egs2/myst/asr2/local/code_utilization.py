import sys
import numpy as np
import matplotlib.pyplot as plt
if __name__=="__main__":
    token_file=sys.argv[1]
    code_array=np.zeros((2000,))
    token_counts=np.zeros((2000,))
    base = ord("一")

    with open(token_file,'r') as f:
        token_scp=f.readlines()
        for line in token_scp:
            code_list=line.strip().split(' ')[-1]
            for code in code_list:
                token=ord(code)-base
                if code_array[token]==0:
                    code_array[token]=1
                token_counts[token]+=1
        code_num=np.sum(code_array)
        code_utilization=float(code_num)/float(2000)
        print("code_utilization: "+str(code_utilization))

        num_tokens=2000
        # 绘制条形图
        token_ids = np.arange(num_tokens)
        token_counts=np.sort(token_counts)[::-1]
        plt.figure(figsize=(16, 6))  # 设置画布大小
        plt.bar(token_ids, token_counts, color='skyblue', edgecolor='blue', width=0.8)

        # 设置图表标题和坐标轴标签
        plt.title("Token ID Frequency Distribution", fontsize=16)
        plt.xlabel("Token ID", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)

        # 设置横轴范围
        plt.xlim(-1, num_tokens)

        # 显示图表
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线
        plt.show()

        output_path = "/data/mohan/workdir/espnet/egs2/myst/asr2/local/code_distribution_ts_kmeans_myst_dev.png"  # 保存路径和文件名
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 文件