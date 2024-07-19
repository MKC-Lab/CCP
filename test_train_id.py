# — coding: utf-8 –
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
filepath = "/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/train_instruction_id.txt"
vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath)
# vs_path = "/home/zzu_zxw/zjl_data/ChatGLM-6B/langchain-ChatGLM/vector_store/train_instruction_id_FAISS_20240307_192537"

def evaluate_and_save_result(test_dataset, output_file_path):
    start_index = 0
    with open(output_file_path, "a", encoding="utf-8") as file:
        for i, data in enumerate(test_dataset[start_index:], start=start_index):
            query = data["instruction"]
            history = []
            ids = local_doc_qa.get_context_based_answer(query=query, vs_path=vs_path)

            result = {
                "instruction": data["instruction"],
                "input": "",
                "output": data["output"],
                "context": ids
            }
            json.dump(result, file, ensure_ascii=False)
            file.write('\n')

def load_test_dataset():
    test_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/all_output.json'
    test_file = open(test_file_path, "r")
    dataset = []
    for line in test_file.readlines():
        record = json.loads(line)
        dataset.append(record)
    return dataset

if __name__ == "__main__":
    test_dataset = load_test_dataset()
    output_file_path = '/home/zzu_zxw/zjl_data/KnowPAT/zjl/chatzhuanzhi/data/all_output_context=7.json'
    evaluate_and_save_result(test_dataset, output_file_path)

