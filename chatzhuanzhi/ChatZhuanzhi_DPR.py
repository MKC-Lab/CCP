from pprint import pprint
import faiss
import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm
from dpr_executor import DPRExecutor
import os

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    executor = DPRExecutor()
    # checkpoint_path = "/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/Experiments/TriviaQA_DPR_FullCorpus_2/train/saved_model/model_5.ckpt"
    checkpoint_path="/home/zzu_zxw/zjl_data/Retrieval-Augmented-Visual-Question-Answering/Experiments/Zhuanzhi_DPR_FullCorpus_1/train/saved_model/model_4.ckpt"
    checkpoint = torch.load(checkpoint_path)
    executor.load_state_dict(checkpoint['state_dict'])

    # 示例用户输入
    user_input = "推荐编译器相关的文章(论文)"

    # 示例包含段落的JSON文件
    passages_json = {
        'id2doc': {
            '0': '推荐投资相关的文章(论文):{4eb50eabfe9a665a7d3b79d529da8af1}',
            '1': '推荐编译器相关的文章(论文):{baf01c39449e31a636adb1d25f36f21c}',
            '2': '推荐编译器相关的文章(论文):{baf01c39449e31a636adb1d25f36f21c}',
            '3': '推荐编程工具相关的文章(论文):{32d8dd843cf185842837852599fa32c8}',
            '4': '推荐兵棋推演相关的文章(论文)',
            '5': '推荐知识图谱的文章(论文)',
            # '6': '推荐分子相关的文章(论文):{704e153d91303076ee8dc79cebc36b52}',
        },
    }



    # 读取txt文件内容并构建passages_json
    # txt_files = [
    #     "/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/src/chatzhuanzhi/ID/ArticleID.txt",
    #     "/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/src/chatzhuanzhi/ID/ConnectID.txt",
    #     # "/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/src/chatzhuanzhi/ID/FundID.txt",
    #     # "/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/src/chatzhuanzhi/ID/TopicID.txt",
    # ]
    # passages_json = {'id2doc': {}}
    # passage_id = 0
    # for i, file_path in enumerate(txt_files):
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         lines = file.readlines()
    #         for j, line in enumerate(lines):
    #             content = line.strip() # 确保每个文件的行都有唯一的id
    #             passages_json['id2doc'][passage_id] = content
    #             passage_id = passage_id + 1


    sample = EasyDict({'question': user_input, })
    questions = sample.question
    input_data = EasyDict({'text_sequence': ['<BOQ>' + questions + '<EOQ>']})
    text_sequences = input_data.pop('text_sequence')
    encoding = executor.tokenizer(text_sequences, padding='longest', max_length=512, truncation=True, return_tensors="pt")
    input_data.update({'input_ids': encoding.input_ids, 'attention_mask': encoding.attention_mask,'input_text_sequences': text_sequences, })
    batched_data = EasyDict({'questions': [questions], })
    batched_data.update(input_data)

    test_batch = EasyDict({
        'input_ids': batched_data['input_ids'].to(executor.device),
        'attention_mask': batched_data['attention_mask'].to(executor.device),
    })
    # batch_size x hidden_states
    query_emb = executor.model.generate_query_embeddings(**test_batch)  # model是Retriver
    data_to_return = {'query_emb': query_emb}

    query_embeddings = []  # 存储查询嵌入
    query_embeddings.append(data_to_return['query_emb'])  # 提取每个验证步骤中的查询嵌入
    query_embeddings = torch.cat(query_embeddings, dim=0)  # 拼接查询嵌入向量

    passage_id2doc = passages_json["id2doc"]
    n_items = len(passage_id2doc)
    hidden_size = query_embeddings.shape[1]
    i_batch_size = 1
    n_item_batchs = n_items // i_batch_size

    passage_index2id = {index: passage_id for index, passage_id in enumerate(passage_id2doc.keys()) if index < n_items}
    passage_contents = []
    for passage_id in passage_id2doc.keys():  # 将每个段落的内容转化为文本序列，并存储在 passage_contents 中
        passage = EasyDict(passage_content=passage_id2doc[passage_id])
        data_collection = []
        return_dict = EasyDict(text_sequence="", )
        return_dict.text_sequence = passage.passage_content
        # return_dict.text_sequence = ' '.join(['<BOK>'] + [passage.passage_content] + ['<EOK>'])
        data_collection.append(return_dict)
        processed_data = EasyDict()
        for data_entry in data_collection:
            for key, value in data_entry.items():
                processed_data[key] = value
        passage_contents.append(processed_data.text_sequence)

    item_embeddings = []
    for i_batch_id in tqdm(range(n_item_batchs)):  # 遍历段落批次
        i_start = i_batch_id * i_batch_size
        i_end = min((i_batch_id + 1) * i_batch_size, n_items)
        passage_contents_batch = passage_contents[i_start:i_end]
        item_encoding = executor.decoder_tokenizer(passage_contents_batch,
                                               padding='longest',
                                               max_length=512,
                                               truncation=True,
                                               return_tensors="pt")
        # 获取段落的输入ID和注意力掩码
        item_input_ids, item_attention_mask = item_encoding.input_ids, item_encoding.attention_mask
        test_batch = EasyDict({
            'input_ids': item_input_ids.to(executor.device),
            'attention_mask': item_attention_mask.to(executor.device),
        })
        # batch_size x hidden_states
        item_emb = executor.model.generate_item_embeddings(**test_batch)
        for x in item_emb:  ## 将生成的段落嵌入添加到列表中
            item_embeddings.append(x.cpu().detach().numpy())

        # index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)  # 使用HNSWFlat索引（近似最近邻搜索）
        index = faiss.IndexFlatIP(hidden_size)#全搜索

    Ks = [1, 5]
    item_embeddings = np.stack(item_embeddings, 0)

    np.save("/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/src/chatzhuanzhi/ID/vectors.npy", item_embeddings)
    vectors = np.load("/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/src/chatzhuanzhi/ID/vectors.npy")

    index.add(vectors)  # 向faiss索引中添加段落嵌入向量
    faiss.write_index(index, "/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/src/chatzhuanzhi/ID/faiss_index.faiss")
    loaded_index = faiss.read_index("/home/zzu_zxw/zjl_data/RRetrieval-Augmented-Visual-Question-Answering/src/chatzhuanzhi/ID/faiss_index.faiss")
    search_res = loaded_index.search(query_embeddings.detach().cpu().numpy(), k=max(Ks))

    for query_id, return_scores, return_passage_index in zip(range(len(query_embeddings)), search_res[0],search_res[1]):
        top_ranking_passages = [{
            'passage_index': i,
            'passage_id': passage_index2id[i],
            'content': passage_contents[i],
            'score': float(return_scores[index]),
        } for index, i in enumerate(return_passage_index)]
    pprint(top_ranking_passages)

if __name__ == '__main__':
    main()
