from pprint import pprint
import numpy as np
import logging

from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

logger = logging.getLogger(__name__)
from tqdm import tqdm
from easydict import EasyDict
import torch
from chatzhuanzhi.retriever_dpr import RetrieverDPR
import pytorch_lightning as pl
import faiss

class DPRExecutor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        tokenizer_path = "/home/zzu_zxw/zjl_data/Retrieval-Augmented-Visual-Question-Answering/src/models/facebook/dpr-question_encoder-single-nq-base"
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(tokenizer_path)
        self.SPECIAL_TOKENS = {"additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"]}
        self.SPECIAL_TOKENS['additional_special_tokens'] = self.tokenizer.additional_special_tokens + self.SPECIAL_TOKENS['additional_special_tokens']
        self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
        # 如果指定，则加载第二个tokenizer
        tokenizer_path = "/home/zzu_zxw/zjl_data/Retrieval-Augmented-Visual-Question-Answering/src/models/facebook/dpr-ctx_encoder-single-nq-base"
        self.decoder_tokenizer = DPRContextEncoderTokenizer.from_pretrained(tokenizer_path)
        self.DECODER_SPECIAL_TOKENS = {"additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],}
        self.DECODER_SPECIAL_TOKENS['additional_special_tokens'] = self.decoder_tokenizer.additional_special_tokens + self.DECODER_SPECIAL_TOKENS['additional_special_tokens']
        self.decoder_tokenizer.add_special_tokens(self.DECODER_SPECIAL_TOKENS)

        self.model = RetrieverDPR()#初始化RetrieverDPR模型
        self.model.resize_token_embeddings(len(self.tokenizer), len(self.decoder_tokenizer))#调整Token Embeddings大小：将标记嵌入的大小调整为分词器的标记数量，以确保嵌入的维度与分词器的词汇表大小匹配。这是为了与分词器一起使用模型。

    def test_step(self, sample_batched, batch_idx):#方法用于执行测试步骤，并在每个epoch结束后记录测试结果。
        return self._compute_query_embeddings_step(sample_batched, batch_idx)

    def test_epoch_end(self, validation_step_outputs):#方法用于执行测试步骤，并在每个epoch结束后记录测试结果。
        log_dict = self.evaluate_outputs(validation_step_outputs)#跳转到evaluate_outputs()
        pprint(log_dict)
        return log_dict
    
    def test_dataloader(self):#用于返回测试数据的数据加载器
        return self.test_data_loader #跳转到okvqa_datasets.py的__len__

    def _compute_query_embeddings_step(self, sample_batched, batch_idx):
        test_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
        })
        # batch_size x hidden_states
        query_emb = self.model.generate_query_embeddings(**test_batch)
        data_to_return = {
            'btach_idx': batch_idx,
            'query_emb': query_emb,
        }
        return data_to_return#跳转到test_epoch_end()

    def evaluate_outputs(self, step_outputs, mode='test'):
        query_embeddings = [] # 存储查询嵌入
        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])#提取每个验证步骤中的查询嵌入
        query_embeddings = torch.cat(query_embeddings, dim=0)# 拼接查询嵌入向量

        passage_id2doc = self.data_loader.data.passages.id2doc
        n_items = len(passage_id2doc)
        hidden_size = query_embeddings.shape[1]
        i_batch_size = 1 # 批处理大小
        n_item_batchs = n_items // i_batch_size  # 计算项目批次的数量

        # 创建段落ID的索引
        passage_index2id = {index:passage_id for index, passage_id in enumerate(passage_id2doc.keys()) if index < n_items}
        passage_contents = []
        for passage_id in passage_id2doc.keys():#将每个段落的内容转化为文本序列，并存储在 passage_contents 中
            sample = EasyDict(passage_content=passage_id2doc[passage_id])

            data_collection = []
            return_dict = EasyDict(text_sequence="", )
            return_dict.text_sequence = ' '.join(['<BOK>'] + [sample.passage_content] + ['<EOK>'])
            data_collection.append(return_dict)
            processed_data = EasyDict()
            for data_entry in data_collection:
                for key, value in data_entry.items():
                    processed_data[key] = value
            passage_contents.append(processed_data.text_sequence)

        item_embeddings = []
        for i_batch_id in tqdm(range(n_item_batchs)):# 遍历段落批次
            i_start = i_batch_id * i_batch_size
            i_end = min((i_batch_id + 1) * i_batch_size, n_items)
            # 获取当前批次的段落内容
            passage_contents_batch = passage_contents[i_start:i_end]
            item_encoding = self.decoder_tokenizer(passage_contents_batch,
                                padding='longest',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
            # 获取段落的输入ID和注意力掩码
            item_input_ids, item_attention_mask = item_encoding.input_ids, item_encoding.attention_mask
            test_batch = EasyDict({# 创建一个包含段落输入的字典
                'input_ids': item_input_ids.to(self.device),
                'attention_mask': item_attention_mask.to(self.device),
            })
            # batch_size x hidden_states
            item_emb = self.model.generate_item_embeddings(**test_batch)
            for x in item_emb: ## 将生成的段落嵌入添加到列表中
                item_embeddings.append(x.cpu().detach().numpy())

            index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)#使用HNSWFlat索引（近似最近邻搜索）
            # index = faiss.IndexFlatIP(hidden_size)#全搜索

        # 获取配置中的Ks值，表示返回的最近邻数目
        Ks = [1, 5]
        item_embeddings = np.stack(item_embeddings, 0)
        index.add(item_embeddings)#向faiss索引中添加段落嵌入向量
        search_res = index.search(query_embeddings.cpu().numpy(), k=max(Ks))#使用faiss索引进行查询，返回64个问题最近邻的100个段落的匹配得分和对应索引

        for query_id, return_scores, return_passage_index in zip(range(len(query_embeddings)), search_res[0], search_res[1]):
            top_ranking_passages = [{
                'passage_index': i,
                'passage_id': passage_index2id[i],
                'content': passage_contents[i],
                'score': float(return_scores[index]),
                } for index, i in enumerate(return_passage_index)]#对于问题所匹配的100个段落以及得分信息

        return top_ranking_passages # 返回评估结果字典 跳转回validation_epoch_end()  测试时跳转回跳转回test_epoch_end()