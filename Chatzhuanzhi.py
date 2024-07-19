import nltk #NLTK是一个流行的自然语言处理（NLP）库，提供了许多用于处理文本和执行各种NLP任务的功能和工具。
import gradio as gr #Gradio是一个用于构建交互式界面的Python
import models.shared as shared
from configs.model_config import * #导入配置文件参数
from chains.local_doc_qa import LocalDocQA#导入了一个名为LocalDocQA的类
from models.loader.args import parser#导入参数
from models.loader import LoaderCheckPoint
# NLTK_DATA_PATH='/home/zzu_zxw/zjl_data/ChatGLM-6B/langchain-ChatGLM/nltk_data'
# nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

args = None
args = parser.parse_args()#解析命令行参数
args_dict = vars(args)#参数转为字典形式
shared.loaderCheckPoint = LoaderCheckPoint(args_dict)#加载自定义 model CheckPoint
llm_model_ins = shared.loaderLLM()  # 加载模型  #用于初始化并返回LLM的实例对象。
llm_model_ins.history_len = LLM_HISTORY_LEN
local_doc_qa = LocalDocQA()# 创建实例local_doc_qa，LocalDocQA 是一个本地文档问答模型的类。这个类封装了问答系统的核心逻辑，包括文档加载、嵌入模型处理、相似性搜索、提示生成和答案生成等功能，以提供基于本地文档的问答服务。
local_doc_qa.init_cfg(llm_model=llm_model_ins, #llm_model指定的语言模型对象，即用于回答问题的语言模型
                      embedding_model=EMBEDDING_MODEL,#指定的嵌入模型名称，用于将文档转换为向量表示。为：'text2vec'
                      embedding_device=EMBEDDING_DEVICE,#指定的嵌入模型设备，即将嵌入模型加载到哪个设备上进行计算。
                      top_k=VECTOR_SEARCH_TOP_K)#指定的相似性搜索结果返回的前 K 个文档。

# filepath = "content/zhuanzhi/triple_zhuti_lunwen.txt"
filepath = "content/zhuanzhi/ID"
# filepath = "content/zhuanzhi/aaa.txt"
#该方法的作用是从文件加载一组文档，并使用FAISS库创建一个向量存储。向量存储用于基于文档嵌入的高效相似性搜索。

#init_knowledge_vector_store方法返回一个向量存储的路径vs_path，
#以及一个空列表（由于在代码中使用了占位符_，表示不需要接收该返回值）。
#你可以使用返回的向量存储路径vs_path来进行后续的相似性搜索操作或其他相关操作。
# vs_path = "/home/zzu_zxw/zjl_data/ChatGLM-6B/langchain-ChatGLM/vector_store/TopicID_FAISS_20230706_182112"#文章加主题加推荐
# vs_path = "/home/zzu_zxw/zjl_data/ChatGLM-6B/langchain-ChatGLM/vector_store/TopicID_FAISS_20230707_155126"#文章有资源链接和下载链接
vs_path = "/home/zzu_zxw/zjl_data/ChatGLM-6B/langchain-ChatGLM/vector_store/TopicID_FAISS_20230707_215737"
# vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath,vs_path)
# vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath)


# 加载预训练模型
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('./paraphrase-multilingual-MiniLM-L12-v2/')
def get_answer(query,  history,streaming: bool = STREAMING):
        for response, history in local_doc_qa.get_knowledge_based_answer(
                query=query, vs_path=vs_path, chat_history=history, streaming=streaming):

            # import html
            # prompt = response["source_documents"]
            # prompt_list = prompt.split("}")[:-1]
            # prompt_content = "".join([
            #     f"<details><summary>Source [{i + 1}]</summary><pre>{html.escape(doc)}</pre></details>"
            #     for i, doc in enumerate(prompt_list)
            # ])
            # history[-1][-1] += prompt_content

            import html
            prompt = response["source_documents"]
            prompt_content = f"<details><summary>Prompt</summary><pre>{html.escape(prompt)}</pre></details>"
            history[-1][-1] += prompt_content

            yield history, ""

        logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},history={history}")#打印到控制台


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉ChatZhuanzhi🎉
"""
init_message = f"""欢迎使用 ChatZhuanzhi！"""

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=600)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)

                query.submit(get_answer,
                             [query, chatbot],
                             [chatbot, query])
    demo.load(
        inputs=None,
        outputs=[],
        queue=True,
        show_progress=False,
    )

(demo
 .queue(concurrency_count=3)
 .launch(server_name='127.0.0.1',
         server_port=8888,
         show_api=False,
         share=True,
         inbrowser=False))
