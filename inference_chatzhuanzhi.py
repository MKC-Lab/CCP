# — coding: utf-8 –
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import models.shared as shared
from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
from models.loader.args import parser
from models.loader import LoaderCheckPoint

args = None
args = parser.parse_args()
args_dict = vars(args)
shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
llm_model_ins = shared.loaderLLM()
llm_model_ins.history_len = LLM_HISTORY_LEN
local_doc_qa = LocalDocQA()
local_doc_qa.init_cfg(llm_model=llm_model_ins,
                      embedding_model=EMBEDDING_MODEL,
                      embedding_device=EMBEDDING_DEVICE,
                      top_k=VECTOR_SEARCH_TOP_K)
filepath = "content/zhuanzhi/ID"
vs_path = "/home/zzu_zxw/zjl_data/ChatGLM-6B/langchain-ChatGLM/vector_store/TopicID_FAISS_20230707_215737"

with open('/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/train.json', 'r',encoding='utf-8') as file:
    train_dataset = json.load(file)
knowledge_file_path = "/home/zzu_zxw/zjl_data/KnowPAT/zjl/all_output_K=3.json";
test_file = open(knowledge_file_path, "r")
knowledge_datasets = []
for line in test_file.readlines():
    record = json.loads(line)
    knowledge_datasets.append(record)

def evaluate_and_save_result(test_dataset, output_file_path):
    start_index = 0
    with open(output_file_path, "a", encoding="utf-8") as file:
        for i, data in enumerate(test_dataset[start_index:], start=start_index):
            query = data["instruction"]
            answer = data["output"]
            train_contexts = []
            context_ids = data["context"]
            for context_id in context_ids:
                train_context = train_dataset[context_id - 1]
                train_contexts.append(train_context)
            history = []
            response_generator = local_doc_qa.get_knowledge_based_answer(query=query, vs_path=vs_path, chat_history=history,train_context=train_contexts,knowledge_dataset=knowledge_datasets)

            for response, _ in response_generator:
                result = {
                    "question": data["instruction"],
                    "answer": answer,
                    "predict": response["result"]
                }
                json.dump(result, file, ensure_ascii=False)
                file.write('\n')
                break

def load_test_dataset():
    test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/all_output_context=7.json'
    test_file = open(test_file_path, "r")
    dataset = []
    for line in test_file.readlines():
        record = json.loads(line)
        dataset.append(record)
    return dataset

if __name__ == "__main__":
    test_dataset = load_test_dataset()
    output_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/CCP_K=3_N=5_result.json'
    evaluate_and_save_result(test_dataset, output_file_path)
