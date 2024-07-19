#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  实现bleu，支持计算中文和英文
# * create time :  2018/5/11下午3:47
# * file name   :  bleu.py
import json
import os
import argparse
import math
from rouge_chinese import Rouge
import jieba


def read_data(candidate_filename, reference_file):
    """
    读取候选文件和参考文件
    :param candidate_filename: 候选文件
    :param reference_file: 参考文件，可以为文件夹，或者是单个的文件
    :return:
    """
    candidates = []
    references = []
    with open(candidate_filename, "r", encoding="utf-8") as fr:
        for line in fr:
            candidates.append(line.strip())
    if ".txt" in reference_file:
        with open(reference_file, "r", encoding="utf-8") as fr:
            reference = []
            for line in fr:
                reference.append(line.strip())
            references.append(reference)
    else:
        for root, _, files in os.walk(reference_file):
            for file in files:
                reference = []
                with open(os.path.join(root, file), "r", encoding="utf-8") as fr:
                    for line in fr:
                        reference.append(line.strip())
                references.append(reference)
    return candidates, references


def calculate_ngram(candidates, references, n, language):
    count_clip = 0
    count = 0
    for index, candidate in enumerate(candidates):
        references_list = lines2dic(references, index, n, language)
        if language == "en":
            words = candidate.split()
        else:
            words = candidate
        limit = len(words) - n + 1
        candidate_dic = {}
        for i in range(limit):
            key = " ".join(words[i: i+n]).lower() if language == "en" else words[i: i+n]
            if key in candidate_dic.keys():
                candidate_dic[key] += 1
            else:
                candidate_dic[key] = 1
        count_clip += clip(candidate_dic, references_list)
        count += limit
    if count_clip == 0:
        pr = 0
    else:
        pr = float(count_clip) / count
    return pr


def brevity_penalty(candidates, references, language):
    c = 0
    r = 0
    for index, candidate in enumerate(candidates):
        c_length = len(candidate.split()) if language == "en" else len(candidate)
        reference_index = [reference[index] for reference in references]
        r_lengths = [len(r.split()) if language == "en" else len(r) for r in reference_index]
        c += c_length
        r += match_reference(c_length, r_lengths)
    if c > r:
        bp = 1
    elif c == 0:
        bp = 1
    else:
        bp = math.exp(1 - float(r) / c)
    return bp


def match_reference(candidate_len, reference_lens):
    """
    计算当c<=r时，最佳匹配的r的长度
    :param candidate_len:
    :param reference_lens:
    :return:
    """
    best_len = abs(reference_lens[0] - candidate_len)
    best_ref = reference_lens[0]
    for length in reference_lens:
        if abs(length - candidate_len) < best_len:
            best_len = abs(length - candidate_len)
            best_ref = length
    return best_ref


def clip(candidate, references):
    count = 0
    for cand in candidate.keys():
        cand_value = candidate[cand]
        max_count = 0
        for reference in references:
            if cand in reference.keys():
                max_count = max(reference[cand], max_count)
        count += min(max_count, cand_value)
    return count


def lines2dic(references, index, n, language):
    reference_list = []
    for reference in references:
        reference_dic = {}
        line = reference[index]
        if language == "en":
            words = line.split()
        else:
            words = line
        limit = len(words) - n + 1
        for i in range(limit):
            key = " ".join(words[i: i+n]).lower() if language == "en" else words[i: i+n]
            if key in reference_dic.keys():
                reference_dic[key] += 1
            else:
                reference_dic[key] = 1
        reference_list.append(reference_dic)
    return reference_list


def geometric_mean(precisions):
    return math.exp(sum([math.log(p) if p != 0 else -math.inf for p in precisions]) / len(precisions))


def bleu(candidate, references, language):
    precisions = []
    for i in range(1, 5):
        pr = calculate_ngram(candidate, references, i, language)
        precisions.append(pr)
    bp = brevity_penalty(candidate, references, language)
    return [pr * bp for pr in precisions]
    # return geometric_mean(precisions) * bp

def rouge(candidate, references):
    rouge = Rouge()
    hypothesis = ' '.join(jieba.cut(candidate[0]))
    reference_strings = [" ".join(reference) for reference in references]
    reference = ' '.join(jieba.cut(reference_strings[0]))
    # Rouge calculation
    scores = rouge.get_scores(hypothesis, reference)

    # Extracting Rouge-1, Rouge-2, and Rouge-L scores
    rouge_1_score = scores[0]["rouge-1"]["f"]
    rouge_2_score = scores[0]["rouge-2"]["f"]
    rouge_l_score = scores[0]["rouge-l"]["f"]

    return [rouge_1_score, rouge_2_score, rouge_l_score]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLEU calculate")
    args = parser.parse_args()
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/Baichuan_result_100.json'
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/chatglm2_result_100.json'
    test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/CCP_result.json'
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/chatzhuanzhi_congnition5_context_K=3,N=1_result.json'
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/CCP_K=3_N=1_result100.json'  # 1100
    #
    # with open(test_file_path, 'r', encoding='utf-8') as file:
    #     data = json.load(file)
    test_file = open(test_file_path, "r")
    data = []
    for line in test_file.readlines():
        record = json.loads(line)
        data.append(record)
    # 初始化累计分数
    total_bleu_1 = 0
    total_bleu_2 = 0
    total_bleu_3 = 0
    total_bleu_4 = 0

    # 初始化累计分数
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_l = 0

    for entry in data:
        reference = [[entry['answer']]]
        candidate = [entry['predict']]
        lang = "ch"
        # candidate, references = read_data(can_file, ref_file)
        bleu_score = bleu(candidate, reference, lang)
        total_bleu_1 += bleu_score[0]
        total_bleu_2 += bleu_score[1]
        total_bleu_3 += bleu_score[2]
        total_bleu_4 += bleu_score[3]

        rouge_scores = rouge(candidate, reference)
        total_rouge_1 += rouge_scores[0]
        total_rouge_2 += rouge_scores[1]
        total_rouge_l += rouge_scores[2]

    # 计算平均分数
    average_bleu_1 = round(total_bleu_1 / len(data) * 100, 2)
    average_bleu_2 = round(total_bleu_2 / len(data) * 100, 2)
    average_bleu_3 = round(total_bleu_3 / len(data) * 100, 2)
    average_bleu_4 = round(total_bleu_4 / len(data) * 100, 2)

    # 计算平均分数
    average_rouge_1 = round(total_rouge_1 / len(data) * 100, 2)
    average_rouge_2 = round(total_rouge_2 / len(data) * 100, 2)
    average_rouge_l = round(total_rouge_l / len(data) * 100, 2)

    print("BLEU-1:", average_bleu_1)
    print("BLEU-2:", average_bleu_2)
    print("BLEU-3:", average_bleu_3)
    print("BLEU-4:", average_bleu_4)

    print("Rouge-1:", average_rouge_1)
    print("Rouge-2:", average_rouge_2)
    print("Rouge-L:", average_rouge_l)
