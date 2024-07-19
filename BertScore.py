# — coding: utf-8 –
import argparse
import json
from bert_score import score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLEU calculate")
    args = parser.parse_args()
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/llama_result.json'
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/nlpcc2016/chatzhuanzhi_nlpcc2016_content_result.json'
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/chatzhuanzhi_congnition5_context_result.json'
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/chatzhuanzhi_congnition5_context_K=3,N=3_result.json'
    test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/CCP_K=3_N=5_result.json'
    # test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/gpt3.5_result.json'
    # with open(test_file_path, 'r', encoding='utf-8') as file:
    #     dataset = json.load(file)
    test_file = open(test_file_path, "r")
    dataset = []
    for line in test_file.readlines():
        record = json.loads(line)
        dataset.append(record)
    total_P = 0
    total_R = 0
    total_F1 = 0
    for entry in dataset:
        # cands = ['我们都曾经年轻过，虽然我们都年少，但还是懂事的', '我们都曾经年轻过，虽然我们都年少，但还是懂事的']
        # refs = ['虽然我们都年少，但还是懂事的', '我们都曾经年轻过，虽然我们都年少，但还是懂事的']
        reference = [entry['answer']]
        candidate = [entry['predict']]
        P, R, F1 = score(candidate, reference,model_type="bert-base-chinese",lang="zh", verbose=True)
        total_P += P
        total_R += R
        total_F1+= F1
        print("reference:", reference)
        print("candidate:", candidate)
        print("P:", P)
        print("R:", R)
        print("F1:", F1)

    # 计算平均分数
    average_P = round(total_P.item() / len(dataset) * 100, 2)
    average_R = round(total_R.item() / len(dataset) * 100, 2)
    average_F1 = round(total_F1.item() / len(dataset) * 100, 2)

    print("average_P:", average_P)
    print("average_R:", average_R)
    print("average_F1:", average_F1)

    # 存储最终结果到文件
    result_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/nlpcc2016/chatzhuanzhi_nlpcc2016__context_result_bertscore.json'
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump({
            "average_P": average_P,
            "average_R": average_R,
            "average_F1": average_F1,
        }, result_file, ensure_ascii=False, indent=2)