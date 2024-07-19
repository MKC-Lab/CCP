#用于加载和处理Hugging Face的预训练语言模型的嵌入
import json
import re
from easydict import EasyDict
import faiss
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#用于创建和管理基于向量的文档嵌入的高效相似性搜索。
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from configs.model_config import *
from pymongo import MongoClient
import datetime
from textsplitter import ChineseTextSplitter#用于将中文文本分段。
from typing import List, Tuple, Dict
from langchain.docstore.document import Document
import numpy as np
from utils import torch_gc
from tqdm import tqdm
from pypinyin import lazy_pinyin
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from models.base import (BaseAnswer,AnswerResult)
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import models.shared as shared
from agent import bing_search
from langchain.docstore.document import Document
from functools import lru_cache


# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    return hash(self.model_name)

#通过修改__hash__方法，将HuggingFaceEmbeddings类变为可哈希的，以便后续进行缓存。
HuggingFaceEmbeddings.__hash__ = _embeddings_hash


# will keep CACHED_VS_NUM of vector store caches
#load_vector_store: 加载向量存储，使用FAISS库创建向量存储，用于基于文档嵌入的高效相似性搜索。
@lru_cache(CACHED_VS_NUM)
def load_vector_store(vs_path, embeddings):
    return FAISS.load_local(vs_path, embeddings)

#tree返回指定目录下所有文件的完整路径和文件名列表。
def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]

#load_file根据文件路径加载文件，包括.md、.txt、.pdf、.jpg和.png等类型的文件。
def load_file(filepath, sentence_size=SENTENCE_SIZE):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")#加载md文件为一段文本
        docs = loader.load()#document list中只有一个元素（一段文本）里面的属性有两种(page_content是原文和metadata是文件路径和页数等)
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)#TextLoader类用于加载文本文件，并提供了自动检测编码的功能。
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)#进行文本分隔，sentence_size参数用于指定每个句子的最大长度。
        docs = loader.load_and_split(textsplitter)#可以将文本文件加载并进行分割，返回一个包含分割后文本的列表，每个文本段落或句子被分割成独立的文本单元，用于后续的处理和向量化操作。
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    write_check_file(filepath, docs)
    return docs#返回每行一个Document的list列表（包含多个Document）


def write_check_file(filepath, docs):#将加载的文件和文档信息写入一个检查文件，以便后续进行验证和调试。
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")#folder_path='content/zhuanzhi/tmp_files'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')#fp='content/zhuanzhi/tmp_files/load_file.txt'
    with open(fp, 'a+', encoding='utf-8') as fout:#以追加模式打开该文件。
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))#先写入文件路径和文档列表的长度信息
        fout.write('\n')
        for i in docs:#遍历文档列表，将每个文档转换为字符串形式并逐行写入文件。
            fout.write(str(i))
            fout.write('\n')
        fout.close()

#generate_prompt: 根据查询和相关文档生成提示模板。
def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


# seperate_list将列表中连续的数字分隔成子列表。
def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

#similarity_search_with_score_by_vector基于query向量搜索相似的文档。
def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
) -> List[Tuple[Document, float]]:
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)#用索引对象self.index对输入向量进行相似性搜索，返回相似文本的得分和索引。
    docs = []
    id_set = set()
    store_len = len(self.index_to_docstore_id)#表示了索引与文档存储的映射关系中索引的总数
    for j, i in enumerate(indices[0]):
        if i == -1 or 0 < self.score_threshold < scores[0][j]:#如果索引为 -1（表示没有匹配），或者匹配的得分超过了设定的阈值，就跳过继续处理下一个结果。
            # This happens when not enough docs are returned.
            continue
        _id = self.index_to_docstore_id[i]#最相似的句子的id号
        doc = self.docstore.search(_id)#拿到最相似的一个句子
        self.chunk_conent = False
        if not self.chunk_conent:#如果 chunk_conent 为 False，表示不需要对文本进行分块处理。
            if not isinstance(doc, Document):#会判断 doc 是否为有效的文档对象，
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc.metadata["score"] = int(scores[0][j])
            docs.append(doc)
            continue
        id_set.add(i)#先把最相似的添加进最终要返回的
        # docs_len = len(doc.page_content)#最相似的句子的长度
        # for k in range(1, max(i, store_len - i)):#从最相似的文档索引 i 开始向前和向后搜索其他相邻的索引
        #     break_flag = False
        #     for l in [i + k, i ]:#i-k改为i
        #         if 0 <= l < len(self.index_to_docstore_id):#如果索引 l 在有效范围内
        #             _id0 = self.index_to_docstore_id[l] #获取与索引对应的文档 ID _id0
        #             doc0 = self.docstore.search(_id0) #通过文档存储（docstore）获取文档对象 doc0
        #             if docs_len + len(doc0.page_content) > self.chunk_size:#如果将当前文档内容长度 docs_len 加上相邻文档的内容长度后超过了设定的块大小 chunk_size，后边就不要了
        #                 break_flag = True
        #                 break
        #             elif doc0.metadata["source"] == doc.metadata["source"]:#否则，如果相邻文档的来源与最相似文档的来源相同，
        #                 docs_len += len(doc0.page_content)#将相邻文档的内容长度累加到 docs_len
        #                 id_set.add(l)#并将相邻文档的索引 l 添加到集合 id_set 中
        #     if break_flag:#如果在搜索过程中出现了 break_flag 为 True 的情况，表示已经超过了设定的块大小，后面的相邻文档将不再添加。
        #         break
    if not self.chunk_conent:#首先，如果 chunk_conent 为 False，即不需要进行分块处理，则直接返回已获取的文档列表 docs。
        return docs
    if len(id_set) == 0 and self.score_threshold > 0:#如果集合 id_set 为空且设定的得分阈值大于 0，表示没有满足条件的文档，直接返回空列表。
        return []
    id_list = sorted(list(id_set))#对文档索引集合 id_set 进行排序
    id_lists = seperate_list(id_list)#并将排序后的索引列表 id_list 分成若干子列表 id_lists。
    for id_seq in id_lists:#对于每个子列表 id_seq，执行以下操作：
        for id in id_seq:#遍历子列表中的每个索引 id，并根据索引获取相应的文档 ID _id
            if id == id_seq[0]:#如果当前索引是子列表中的第一个索引，则通过文档存储（docstore）获取文档对象 doc。
                _id = self.index_to_docstore_id[id]
                doc = self.docstore.search(_id)
            else:#否则，获取相邻索引对应的文档 ID _id0，并通过文档存储（docstore）获取文档对象 doc0。
                _id0 = self.index_to_docstore_id[id]
                doc0 = self.docstore.search(_id0)
                doc.page_content += " " + doc0.page_content#将相邻文档的内容追加到主文档 doc 的内容后面。
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])#计算当前文档的得分 doc_score，即子列表中所有索引对应的最小得分。
        doc.metadata["score"] = int(doc_score)#将得分转换为整数类型，并将其存储到文档的元数据（metadata）中。
        docs.append(doc)#将文档对象 doc 添加到结果列表 docs 中。
    torch_gc()
    return docs

#search_result2docs(search_results): 将搜索结果转换为文档列表。
def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


class LocalDocQA:
    llm: BaseAnswer = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    #该init_cfg方法初始化系统的配置，包括嵌入模型和设备，以及用于回答问题的语言模型。
    # 它使用 HuggingFaceEmbeddings 类加载指定的嵌入模型。
    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_model: BaseAnswer = None,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm = llm_model
        #model_kwargs：一个字典，包含用于初始化嵌入模型的参数。在这里，只包含一个键值对 {'device': embedding_device}，用于指定将嵌入模型加载到哪个设备上进行计算。
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],#从字典中获取嵌入模型名称
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    #该init_knowledge_vector_store方法从文件或目录加载一组文档，并使用 FAISS 库创建向量存储。
    #向量存储用于基于文档嵌入的高效相似性搜索。
    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],#要加载的文件路径，可以是单个文件路径的字符串或文件路径列表
                                    vs_path: str or os.PathLike = None,#向量存储的路径，默认为None。
                                    sentence_size=SENTENCE_SIZE):#句子的大小，默认为SENTENCE_SIZE。
        #用于记录成功加载的文件和加载失败的文件。
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):#如果filepath是一个字符串
            if not os.path.exists(filepath):#如果文件路径不存在，打印提示信息并返回None。
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):#如果文件路径是一个文件，提取文件名，并尝试加载该文件
                file = os.path.split(filepath)[-1]#file='aaa.txt'
                try:#如果加载成功
                    docs = load_file(filepath, sentence_size)#load_file根据文件路径加载文件
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(filepath)#将文件路径添加到loaded_files列表中
                except Exception as e:#如果加载失败，记录错误信息并返回None。
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):#如果文件路径是一个目录，遍历目录下的所有文件，并尝试加载每个文件的文档。
                docs = []#最终docs列表将包含所有成功加载的文档。以便后续进行向量化和构建向量存储。
                for fullfilepath, file in tqdm(zip(*tree(filepath, ignore_dir_names=['tmp_files'])), desc="加载文件"):
                    try:#如果加载成功，将文件路径添加到loaded_files列表中；
                        docs += load_file(fullfilepath, sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:#如果加载失败，记录错误信息并将文件名添加到failed_files列表中。
                        logger.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logger.info(f"{file}\n")

        else:#如果filepath是一个文件路径列表
            docs = []#最终docs列表将包含所有成功加载的文档。以便后续进行向量化和构建向量存储。
            for file in filepath:#遍历每个文件路径，并尝试加载每个文件的文档
                try:#将文件路径添加到loaded_files列表中
                    docs += load_file(file)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:#如果加载失败，记录错误信息并返回None。
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:#如果成功加载了至少一个文档，进入下一步操作。文本的分隔
            logger.info("文件加载完毕，正在生成向量库")
            #如果指定了向量存储路径vs_path，且该路径存在且包含名为"index.faiss"的文件，
            #说明之前已经创建了向量存储，加载该向量存储，并将新加载的文档添加到其中。
            if vs_path and os.path.isdir(vs_path) and "index.faiss" in os.listdir(vs_path):
                vector_store = load_vector_store(vs_path, self.embeddings)
                # vector_store.add_documents(docs)
                torch_gc()
            #如果不存在指定的向量存储路径vs_path，则根据文件名生成一个默认的向量存储路径，
            #并使用FAISS.from_documents方法将文档转换为向量，并创建一个新的向量存储。
            else:
                if not vs_path:
                    vs_path = os.path.join(VS_ROOT_PATH,
                                           f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
                # vector_store = FAISS.from_documents(docs, self.embeddings) #docs为Document列表 #接受文档列表和嵌入模型作为参数，并返回一个向量库对象。
                torch_gc()#用于执行PyTorch的垃圾回收操作。

            # vector_store.save_local(vs_path)#将向量存储保存到本地
            return vs_path, loaded_files#回向量存储路径vs_path和成功加载的文件列表loaded_files。
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    #该one_knowledge_add方法允许将单个文档添加到现有的向量存储中。
    # 它将标题、内容和内容分段标志作为输入，并将文档附加到向量存储中。
    def one_knowledge_add(self, vs_path, one_title, one_conent, one_content_segmentation, sentence_size):
        try:
            if not vs_path or not one_title or not one_conent:
                logger.info("知识库添加错误，请确认知识库名字、标题、内容是否正确！")
                return None, [one_title]
            docs = [Document(page_content=one_conent + "\n", metadata={"source": one_title})]
            if not one_content_segmentation:
                text_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)#分隔，一段文本有sentence_size个字符
                docs = text_splitter.split_documents(docs)#分割知识库后，docs变成了sentence_size个字符一个list元素
            if os.path.isdir(vs_path) and os.path.isfile(vs_path+"/index.faiss"):
                vector_store = load_vector_store(vs_path, self.embeddings)#构建向量库
                vector_store.add_documents(docs)
            else:
                vector_store = FAISS.from_documents(docs, self.embeddings)  ##docs 为Document列表  #构建向量库vector_store
            torch_gc()
            vector_store.save_local(vs_path)
            return vs_path, [one_title]
        except Exception as e:
            logger.error(e)
            return None, [one_title]


    def get_context_based_answer(self, query, vs_path):
        vector_store = load_vector_store(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size#300。匹配后单段上下文长度（在匹配的文本分句长度前后扩充至CHUNK_SIZE字）
        vector_store.chunk_conent = self.chunk_conent#true 在相似性搜索期间返回块的内容。
        vector_store.score_threshold = self.score_threshold#500
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        print(f"匹配到的为:\n")
        for document in related_docs_with_score:
            print(str(document) + "\n")
        torch_gc()

        ids = []
        for document in related_docs_with_score:
            page_content = document.page_content
            match = re.match(r'^(\d+)', page_content)
            if match:
                index = int(match.group(1))
                print(f"提取到的数字：{index}")
            ids.append(index)

        # for document in related_docs_with_score:
        #     page_content = document.page_content
        #     match = re.search(r'{([^}]*)}', page_content)
        #     if match:
        #         index = match.group(1)
        #         print(f"提取到的数字：{index}")
        #         ids.append(index)

        return ids

    def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):#Chatzhuanzhi
        vector_store = load_vector_store(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size#300。匹配后单段上下文长度（在匹配的文本分句长度前后扩充至CHUNK_SIZE字）
        vector_store.chunk_conent = self.chunk_conent#true 在相似性搜索期间返回块的内容。
        vector_store.score_threshold = self.score_threshold#500
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        print(f"匹配到的为:\n")
        for document in related_docs_with_score:
            print(str(document) + "\n")
        torch_gc()

        target_ids = []
        for document in related_docs_with_score:
            page_content = document.page_content
            target_id = page_content.split("{")[1].split("}")[0]
            target_ids.append(target_id)

        # id去重并保持原有顺序（最相似的在最前）
        unique_target_ids = list(dict.fromkeys(target_ids))
        import search
        documents = search.find_data_by_ids(unique_target_ids)
        content = search.documents_to_content(documents)

        if len(content) > 0:
            prompt = search.generate_prompt(content, query)
        else:
            prompt = query;
        print(f"\n提示模板为:\n {prompt}")
########
        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,streaming=streaming):
            resp = answer_result.llm_output["answer"]#获取生成的回答结果 resp
            history = answer_result.history#以及生成回答时的历史记录 history
            history[-1][0] = query#将查询替换为当前的查询，更新历史记录的最后一个元素的查询内容。
            response = {"query": query,#构建一个包含查询、生成的回答结果和相关的文档信息的字典 response。
                        "result": resp,
                        "source_documents": prompt}#related_docs_with_score改为content

            yield response, history#会返回一个包含答案和相关信息的结果，同时保留更新后的历史记录供下一次调用使用。

    def get_knowledge_k_based_answer(self, query, vs_path, train_context, knowledge_dataset, chat_history=[],
                                   streaming: bool = STREAMING):  # chinamm
        vector_store = load_vector_store(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size  # 300。匹配后单段上下文长度（在匹配的文本分句长度前后扩充至CHUNK_SIZE字）
        vector_store.chunk_conent = self.chunk_conent  # true 在相似性搜索期间返回块的内容。
        vector_store.score_threshold = self.score_threshold  # 500
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        print(f"匹配到的为:\n")
        for document in related_docs_with_score:
            print(str(document) + "\n")
        torch_gc()
        #########知识图谱的提示模板构建
        # content = []
        # for document in related_docs_with_score:
        #     page_content = document.page_content
        #     content.append(page_content)
        # import search
        # if len(content) > 0:
        #     prompt = search.generate_prompt(train_context,content, query)
        earch.generate_prompt(train_context,content, query)
        # print(f"\n提示模板为:\n {prompt}")
        #########
        target_ids = []
        for document in related_docs_with_score:
            page_content = document.page_content
            target_id = page_content.split("{")[1].split("}")[0]
            target_ids.append(target_id)

        for document in knowledge_dataset:
            if query == document["instruction"]:
                target_ids = document["K"]
        # id去重并保持原有顺序（最相似的在最前）
        unique_target_ids = list(dict.fromkeys(target_ids))
        import search
        documents = search.find_data_by_ids(unique_target_ids)
        content = search.documents_to_content(documents)
        # content = []
        if len(content) > 0:
            prompt = search.generate_prompt(train_context, content, query)
        else:
            prompt = search.generate_prompt(train_context, content, query)
        print(f"\n提示模板为:\n {prompt}")
        ########
        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=False):  # 原来streaming=streaming
            resp = answer_result.llm_output["answer"]  # 获取生成的回答结果 resp
            history = answer_result.history  # 以及生成回答时的历史记录 history
            history[-1][0] = query  # 将查询替换为当前的查询，更新历史记录的最后一个元素的查询内容。
            response = {"query": query,  # 构建一个包含查询、生成的回答结果和相关的文档信息的字典 response。
                        "result": resp,
                        "source_documents": prompt}  # related_docs_with_score改为content

            yield response, history  # 会返回一个包含答案和相关信息的结果，同时保留更新后的历史记录供下一次调用使用。

    # query      查询内容
    # vs_path    知识库路径
    # chunk_conent   是否启用上下文关联
    # score_threshold    搜索匹配score阈值
    # vector_search_top_k   搜索知识库内容条数，默认搜索5条结果
    # chunk_sizes    匹配单段内容的连接上下文长度

    #该get_knowledge_based_conent_test方法与get_knowledge_based_answer类似，
    #但返回相关文档和提示字符串，而不是生成答案。它用于测试和调试目的。
    def get_knowledge_based_conent_test(self, query, vs_path, chunk_conent,
                                        score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
                                        vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_size=CHUNK_SIZE):
        vector_store = load_vector_store(vs_path, self.embeddings)#从本地加载向量化知识库
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_conent = chunk_conent
        vector_store.score_threshold = score_threshold
        vector_store.chunk_size = chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=vector_search_top_k)#根据query从向量库中查找
        if not related_docs_with_score:
            response = {"query": query,
                        "source_documents": []}
            return response, ""
        torch_gc()
        prompt = "\n".join([doc.page_content for doc in related_docs_with_score])#从匹配到的相似文本中，构建已知信息
        response = {"query": query,
                    "source_documents": related_docs_with_score}#query和相关内容组成提示模板
        return response, prompt

    #该get_search_result_based_answer方法执行基于搜索结果的问答过程。它将查询和可选的聊天历史记录作为输入，对查询执行Bing 搜索，
    #将搜索结果转换为文档对象，使用搜索结果文档和查询生成提示，并使用语言模型生成答案。它返回生成的答案和更新的聊天历史记录。
    def get_search_result_based_answer(self, query, chat_history=[], streaming: bool = STREAMING):
        results = bing_search(query)
        result_docs = search_result2docs(results)
        prompt = generate_prompt(result_docs, query)

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": result_docs}
            yield response, history

#if用于判断当前模块是否作为主程序运行。如果您在命令行中直接运行这个脚本，那么条件为真，以下代码块将会执行。
#如果这个脚本被其他Python脚本导入并调用，那么条件为假，以下代码块将不会执行。
#在Python中，__name__ 是一个内置变量，用于表示当前模块的名称。
#当一个 Python 脚本被直接执行时，其 __name__ 的值被设置为 "__main__"，表示该模块是主程序入口。
#而当一个 Python 模块被导入到其他模块中时，__name__ 的值被设置为模块的名称，即模块的文件名（不包含文件扩展名）。
if __name__ == "__main__":
    # 初始化消息
    args = None
    args = parser.parse_args(args=['--model-dir', '/media/checkpoint/', '--model', 'chatglm-6b', '--no-remote-model'])

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)#设置llm_model_ins实例的历史记录长度。

    local_doc_qa = LocalDocQA()#创建一个LocalDocQA对象，
    local_doc_qa.init_cfg(llm_model=llm_model_ins)#对local_doc_qa进行初始化配置。
    query = "本项目使用的embedding模型是什么，消耗多少显存"
    vs_path = "/media/gpt4-pdf-chatbot-langchain/dev-langchain-ChatGLM/vector_store/test"
    last_print_len = 0
    # for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
    #                                                              vs_path=vs_path,
    #                                                              chat_history=[],
    #                                                              streaming=True):
    for resp, history in local_doc_qa.get_search_result_based_answer(query=query,
                                                                     chat_history=[],
                                                                     streaming=True):
        print(resp["result"][last_print_len:], end="", flush=True)
        last_print_len = len(resp["result"])
    #根据获取到的resp和source_documents，生成一段包含出处信息的文本，并使用日志记录器logger打印输出。
    source_text = [f"""出处 [{inum + 1}] {doc.metadata['source'] if doc.metadata['source'].startswith("http") 
                   else os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                   # f"""相关度：{doc.metadata['score']}\n\n"""
                   for inum, doc in
                   enumerate(resp["source_documents"])]
    logger.info("\n\n" + "\n\n".join(source_text))
    pass
