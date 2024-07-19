# — coding: utf-8 –
from langchain.chains import llm
from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
from typing import List, Tuple, Dict
from pymongo import MongoClient
from bs4 import BeautifulSoup
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


# Show reply with source text from input document
REPLY_WITH_SOURCE = True


def main():
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()#加载模型
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)

    filepath = "content/zhuanzhi/aaa.txt"
    vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath)

    history = []
    while True:
        query = input("Input your question 请输入问题：")
        # query = "推荐关于强化学习的文章"
        response, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query,vs_path=vs_path,chunk_conent=True,
                                        score_threshold=500,
                                        vector_search_top_k=2, chunk_size=0)
        target = response['source_documents']
        target_ids = []
        for document in target:
            page_content = document.page_content
            target_id = page_content.split("{")[1].split("}")[0]
            target_ids.append(target_id)

        print(target_ids)
        documents = find_data_by_ids(target_ids)
        content = documents_to_content(documents )

        print(f"文章内容为：{content}")

        if len(content) > 0:
            prompt = generate_prompt(content, query)
            print(f"模板为：{prompt}")
        else:
            prompt = query

        last_print_len = 0
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=prompt, history=history,
                                                                   streaming=True):
            resp = answer_result.llm_output["answer"]

            print(resp[last_print_len:], end="", flush=True)
            last_print_len = len(resp)

            history = answer_result.history  # 以及生成回答时的历史记录 history
            history[-1][0] = query  # 将查询替换为当前的查询，更新历史记录的最后一个元素的查询内容。


            # yield response, history  # 会返回一个包含答案和相关信息的结果，同时保留更新后的历史记录供下一次调用使用。
        print("\n")
        print(f"历史记录为：{history}")
        print("\n")




# 根据多个ID查找数据
def find_data_by_ids(ids):
    client = MongoClient("localhost", 8191)
    db = client["zhuanzhi"]
    collection2 = db["document"]
    collection1 = db["topic1.0"]
    data_list = []
    for id in ids:
        if id.startswith("2001") or id.startswith("tp"):#主题表
            data = collection1.find_one({"_id": id})
            if data:
                data_list.append(data)
        else:#document表
            data = collection2.find_one({"_id": id})
            if data:
                data_list.append(data)
    return data_list

def documents_to_content(documents):
    article_info_list = []  # 存储所有文章、基金、主题信息的列表
    for item in documents:
        #是否为主题
        if item["_id"].startswith("2001") or item["_id"].startswith("tp"):
            # 提取字段数据
            name = item.get("name", "暂无信息")
            description = BeautifulSoup(item.get("description", "暂无信息"), "html.parser").get_text().replace("\n","").replace("\r","").replace(" ","")[:400]
            if not description:
                description = "暂无信息"
            alias = item.get("alias")
            alias = "、".join(alias) if alias else "暂无信息"
            childrenTopics = "、".join([topic["name"] for topic in item.get("childrenTopics", [])]) or "暂无信息"
            fatherTopics = "、".join([topic["name"] for topic in item.get("fatherTopics", [])]) or "暂无信息"
            simTopics = "、".join([topic["name"] for topic in item.get("simTopics", [])]) or "暂无信息"
            simPersonTopics = "、".join([topic["name"] for topic in item.get("simPersonTopics", [])]) or "暂无信息"
            sourceLinks = "、".join([f"{link['name']}: {link['url']}" for link in item.get("sourceLinks", [])]) or "暂无信息"

            PROMPT_TEMPLATE = """下面是一个主题的信息：
                {{
                主题的名字：{name}
                主题的别名：{alias}
                主题的子主题：{childrenTopics}
                主题的父主题：{fatherTopics}
                主题的相似主题：{simTopics}
                主题的相关人物：{simPersonTopics}
                主题的资源链接：{sourceLinks}
                主题的描述信息：{description}
                }}
                """.strip()

            # 格式化提示模板
            article_info = PROMPT_TEMPLATE.format(
                name=name,
                alias=alias,
                childrenTopics=childrenTopics,
                fatherTopics=fatherTopics,
                simTopics=simTopics,
                simPersonTopics=simPersonTopics,
                sourceLinks=sourceLinks,
                description = description
            )
            article_info_list.append(article_info)
        #如果是文章或者基金
        else:
            #如果是文章
            if item["source"] == "vip_c":
                topic_keywords = [keyword["word"] for keyword in item["topicKTM"]]
                if not topic_keywords:
                    topic_keywords = "暂无信息"
                abstract = BeautifulSoup(item.get("abstractContent", {}).get("content", ""),"html.parser").get_text().replace("\n", "")[:400]
                if not abstract:
                    abstract = "暂无信息"
                author = item["author"]["name"]
                if not author:
                    author = "暂无信息"
                title = item["title"]
                if not title:
                    title = "暂无信息"
                published_time = item["publishedTime"]
                if not published_time:
                    published_time = ["暂无信息"]
                resourceUrl = item["obj"]["resourceUrl"]
                if not resourceUrl:
                    resourceUrl = "暂无信息"
                PROMPT_TEMPLATE = """下面是一篇文章的信息：
                {{
                文章的题目：{title}
                文章的相关主题：{topic_keywords}
                文章的作者：{author}
                文章的发表时间：{published_time}
                文章的链接：{resourceUrl}
                文章的摘要：{abstract}
                }}
                """.strip()

                # 格式化提示模板
                article_info = PROMPT_TEMPLATE.format(
                    topic_keywords="、".join(topic_keywords),
                    abstract=abstract,
                    author=author,
                    title=title,
                    published_time=published_time,
                    resourceUrl=resourceUrl
                )
                article_info_list.append(article_info)
            # 如果是基金
            if item["source"] == "autofund":
                # 提取字段数据并判断是否为空
                title = item.get("title", "暂无信息")
                if item["topicKTM"] is not None:
                    topic_keywords = "、".join([topic["word"] for topic in item.get("topicKTM", [])]) or "暂无信息"
                else:
                    topic_keywords = "暂无信息"
                key = item.get("key", "暂无信息")
                creatOrorganization = item["obj"].get("creatOrorganization", "暂无信息")
                projectName = item["obj"].get("projectName", "暂无信息")
                competentOrg = item["obj"].get("competentOrg", "暂无信息")
                creator = item["obj"].get("creator", "暂无信息")
                linkmanEmail = item["obj"].get("linkmanEmail", "暂无信息")
                if linkmanEmail is None:#如果有该键值对，但是值为None
                    linkmanEmail = "暂无信息"
                linkmanAddresss = item["obj"].get("linkmanAddresss", "暂无信息")#如果没有该字段，则设置为暂无信息
                if linkmanAddresss is None:#如果有该键值对，但是值为None
                    linkmanAddresss = "暂无信息"
                url = item.get("url", "暂无信息")
                classification = item["obj"].get("classification", "暂无信息")
                proposalDate = item["obj"].get("proposalDate", "暂无信息")
                abstractCn = item["obj"].get("abstractCn", "暂无信息").replace("\n","").replace("\r","").replace(" ","")[:400]

                PROMPT_TEMPLATE = """下面是一个基金项目的信息：
                {{
                基金项目的名称：{title}
                基金项目的关键词：{topic_keywords}
                基金项目的项目编号：{key}
                基金项目的依托单位：{creatOrorganization}
                基金项目的项目类型：{projectName}
                基金项目的委员会：{competentOrg}
                基金项目的负责人：{creator}
                基金项目的负责人邮箱：{linkmanEmail}
                基金项目的负责人单位：{linkmanAddresss}
                基金项目的链接：{url}
                基金项目的项目学科：{classification}
                基金项目的批准年度：{proposalDate}
                基金项目的摘要：{abstractCn}
                }}""".strip()

                # 格式化提示模板
                article_info = PROMPT_TEMPLATE.format(
                    title=title,
                    topic_keywords=topic_keywords,
                    key=key,
                    creatOrorganization=creatOrorganization,
                    projectName=projectName,
                    competentOrg=competentOrg,
                    creator=creator,
                    linkmanEmail=linkmanEmail,
                    linkmanAddresss=linkmanAddresss,
                    url=url,
                    classification=classification,
                    proposalDate=proposalDate,
                    abstractCn=abstractCn
                )
                article_info_list.append(article_info)
    return article_info_list

# def generate_prompt(related_docs: List[str], query: str, prompt_template: str = PROMPT_TEMPLATE) -> str:
#     # context = "\n".join(str(doc) for doc in related_docs)  # 将列表中的字典元素转换为字符串
#     prompt = prompt_template.replace("{question}", query)
        # .replace("{context}", context)
    return prompt
# def generate_prompt(train_context, related_docs: List[str], query: str, prompt_template: str = PROMPT_TEMPLATE) -> str:
#     train_context_question = str(train_context["instruction"])
#     train_context_answer = str(train_context["output"])
#     context = "\n".join(str(doc) for doc in related_docs)  # 将列表中的字典元素转换为字符串
#     prompt = prompt_template.replace("{question}", query).replace("{context}", context)\
#         .replace("{train_context_question}",train_context_question).replace("{train_context_answer}",train_context_answer)
#     return prompt
def generate_prompt(train_context, related_docs: List[str], query: str, prompt_template: str = PROMPT_TEMPLATE) -> str:
    train_context_question1 = str(train_context[0]["instruction"])
    train_context_answer1 = str(train_context[0]["output"])
    train_context_question2 = str(train_context[1]["instruction"])
    train_context_answer2 = str(train_context[1]["output"])
    train_context_question3 = str(train_context[2]["instruction"])
    train_context_answer3 = str(train_context[2]["output"])
    train_context_question4 = str(train_context[3]["instruction"])
    train_context_answer4 = str(train_context[3]["output"])
    train_context_question5 = str(train_context[4]["instruction"])
    train_context_answer5 = str(train_context[4]["output"])
    context = "\n".join(str(doc) for doc in related_docs)  # 将列表中的字典元素转换为字符串
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)\
        .replace("{train_context_question1}",train_context_question1).replace("{train_context_answer1}",train_context_answer1) \
        .replace("{train_context_question2}", train_context_question2).replace("{train_context_answer2}", train_context_answer2) \
        .replace("{train_context_question3}", train_context_question3).replace("{train_context_answer3}", train_context_answer3) \
        .replace("{train_context_question4}", train_context_question4).replace("{train_context_answer4}",train_context_answer4) \
        .replace("{train_context_question5}", train_context_question5).replace("{train_context_answer5}",train_context_answer5)
    return prompt


if __name__ == "__main__":
    main()
