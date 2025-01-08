def calculate_wer(hypothesis, reference):
    hypothesis = hypothesis.lower().replace('<sc>','')
    reference = reference.lower().replace('<sc>','')

    # 将句子分割为单词
    hypothesis_words = hypothesis.split()
    reference_words = reference.split()

    # 计算编辑距离
    distance_matrix = [[0] * (len(reference_words) + 1) for _ in range(len(hypothesis_words) + 1)]
    for i in range(len(hypothesis_words) + 1):
        distance_matrix[i][0] = i
    for j in range(len(reference_words) + 1):
        distance_matrix[0][j] = j

    for i in range(1, len(hypothesis_words) + 1):
        for j in range(1, len(reference_words) + 1):
            if hypothesis_words[i - 1] == reference_words[j - 1]:
                cost = 0
            else:
                cost = 1
            distance_matrix[i][j] = min(distance_matrix[i - 1][j] + 1,  # 删除操作
                                        distance_matrix[i][j - 1] + 1,  # 插入操作
                                        distance_matrix[i - 1][j - 1] + cost)  # 替换操作

    # 计算WER
    wer = distance_matrix[len(hypothesis_words)][len(reference_words)] / len(reference_words) * 100
    return wer

import os
def main():
    file_dir="/apdcephfs_cq10/share_1603164/user/morganshi/workspace/InstructSpeech/inference/continus_asr/test_other"
    result_file = os.path.join(file_dir,"output.txt")
    text_file = os.path.join(file_dir,"target.txt")
    output_file = os.path.join(file_dir,"wer.txt")

    # 读取结果和参考文本
    with open(result_file, "r") as f:
        results = f.readlines()
    with open(text_file, "r") as f:
        references = f.readlines()

    # 计算WER并构建WER和结果的字典
    wer_dict = {}
    for result, reference in zip(results, references):
        result = result.strip()
        reference = reference.strip()
        wer = calculate_wer(result, reference)
        wer_dict[result] = wer

    # 按WER由高到低排序
    sorted_results = sorted(wer_dict.items(), key=lambda x: x[1], reverse=True)

    # 去重并重新编写WER
    unique_results = []
    unique_wers = []
    for result, wer in sorted_results:
        if result not in unique_results:
            unique_results.append(result)
            unique_wers.append(wer)

    # 写入结果到output_file
    with open(output_file, "w") as f:
        for result, wer in zip(unique_results, unique_wers):
            f.write(f"{result} (WER: {wer:.2f}%)\n")
    print("wer:",sum(unique_wers)/len(unique_wers))


if __name__ == "__main__":
    main()
