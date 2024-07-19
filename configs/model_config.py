import torch.cuda
import torch.backends
import os
import logging
import uuid

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"#日志的格式字符串，定义了日志中各个字段的排列方式和格式。
logger = logging.getLogger()#logger是日志记录器对象，通过logging.getLogger()创建，
logger.setLevel(logging.INFO)#并设置了日志级别为logging.INFO，表示只记录INFO级别及以上的日志。
logging.basicConfig(format=LOG_FORMAT)#配置日志的基本设置，包括日志级别和格式。其中，format参数设置为LOG_FORMAT，即上述定义的日志格式。

#embedding_model_dict 是一个字典，其中包含了不同嵌入模型的名称和对应的模型路径。
#通过这个字典，可以根据所需的嵌入模型名称，获取对应的模型路径。
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    # "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec": "../model/text2vec/text2vec-large-chinese",
    "m3e-small": "moka-ai/m3e-small",
    "m3e-base": "moka-ai/m3e-base",
}


# Embedding model name
EMBEDDING_MODEL = "text2vec"#所选用的嵌入式模型的名称。默认为"text2vec"，即使用"text2vec-large-chinese"模型。

#表示嵌入式模型的运行设备。根据CUDA的可用性，它被设置为"cuda"、"mps"或"cpu"。
# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
# llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
#存储了不同LLM模型的名称和相关信息。每个模型都包含了模型名称、预训练模型名称、本地模型路径和提供的服务类型等信息。
llm_model_dict = {
    "chatglm-6b-int4-qe": {
        "name": "chatglm-6b-int4-qe",
        "pretrained_model_name": "THUDM/chatglm-6b-int4-qe",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
    "chatglm-6b-int4": {
        "name": "chatglm-6b-int4",
        "pretrained_model_name": "THUDM/chatglm-6b-int4",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
    "chatglm-6b-int8": {
        "name": "chatglm-6b-int8",
        "pretrained_model_name": "THUDM/chatglm-6b-int8",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": None,
        "provides": "ChatGLM"
    },

    "chatyuan": {
        "name": "chatyuan",
        "pretrained_model_name": "ClueAI/ChatYuan-large-v2",
        "local_model_path": None,
        "provides": None
    },
    "moss": {
        "name": "moss",
        "pretrained_model_name": "fnlp/moss-moon-003-sft",
        "local_model_path": None,
        "provides": "MOSSLLM"
    },
    "vicuna-13b-hf": {
        "name": "vicuna-13b-hf",
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": None,
        "provides": "LLamaLLM"
    },

    # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-chatglm-6b": {
        "name": "chatglm-6b",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "chatglm-6b",
        "local_model_path": None,
        "provides": "FastChatOpenAILLM",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLM"
        "api_base_url": "http://localhost:8000/v1"  # "name"修改为fastchat服务中的"api_base_url"
    },

    # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-vicuna-13b-hf": {
        "name": "vicuna-13b-hf",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": None,
        "provides": "FastChatOpenAILLM",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLM"
        "api_base_url": "http://localhost:8000/v1"  # "name"修改为fastchat服务中的"api_base_url"
    },
}

# LLM 名称
LLM_MODEL = "chatglm-6b" #表示所选用的LLM模型的名称。默认为"chatglm-6b"。
# 如果你需要加载本地的model，指定这个参数--no-remote-model，或者下方参数修改为 `True`
NO_REMOTE_MODEL = True
# 量化加载8bit 模型
LOAD_IN_8BIT = False #表示是否以8位精度加载模型
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False #表示是否以bfloat16精度加载模型
# 本地模型存放的位置
MODEL_DIR = "../model"
# 本地lora存放的位置
LORA_DIR = "../ChatGLM-Tuning/output"

# LLM lora path，默认为空，如果有请直接指定文件夹路径
# LLM_LORA_PATH = ""
LLM_LORA_PATH = "../ChatGLM-Tuning/output"
USE_LORA = True if LLM_LORA_PATH else False

# LLM streaming reponse
STREAMING = True  #表示LLM模型是否使用流式响应

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False #表示是否使用p-tuning-v2 PrefixEncoder

# LLM running device  #表示LLM模型的运行设备。根据CUDA的可用性，它被设置为"cuda"、"mps"或"cpu"。
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#表示向量存储的根路径。默认为当前文件的上一级目录下的"vector_store"文件夹。
#'/home/zzu_zxw/zjl_data/ChatGLM-6B/langchain-ChatGLM/vector_store'
VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

#表示上传内容的根路径。默认为当前文件的上一级目录下的"content"文件夹。
UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"  #修改模板
PROMPT_TEMPLATE = """已知信息：
{context}

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请用自己的知识回答，但不允许在答案中添加编造成分，答案请使用中文。问题是：{question}"""


# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
# PROMPT_TEMPLATE = """请使用中文回答问题：{question}"""


# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
# PROMPT_TEMPLATE = """请根据问答示例上下文，简洁和专业的来回答问题，不允许在答案中添加编造成分，答案请使用中文。
# 上下文示例：
# 问题：{train_context_question1}
# 答案：{train_context_answer1}
# ======
# 问题：{train_context_question2}
# 答案：{train_context_answer2}
# ======
# 问题：{train_context_question3}
# 答案：{train_context_answer3}
# ======
# 问题：{train_context_question4}
# 答案：{train_context_answer4}
# ======
# 问题：{train_context_question5}
# 答案：{train_context_answer5}
# ======
# 问题：{train_context_question6}
# 答案：{train_context_answer6}
# ======
# 问题：{train_context_question7}
# 答案：{train_context_answer7}
# ======
# 你要回答的问题是：{question}"""
# PROMPT_TEMPLATE = """根据已知信息和问答示例上下文，从已知信息中提取答案，简洁和专业的来回答用户问题，不允许编造答案，答案请使用中文。
# 问答示例上下文：
# # ======
# 问题：{train_context_question1}
# 答案：{train_context_answer1}
# # ======
# 问题：{train_context_question2}
# 答案：{train_context_answer2}
# # ======
# 问题：{train_context_question3}
# 答案：{train_context_answer3}
# # ======
# 问题：{train_context_question4}
# 答案：{train_context_answer4}
# # ======
# 问题：{train_context_question5}
# 答案：{train_context_answer5}
# # ======
# 已知信息：
# {context}
# 问题：{question}
# 答案："""

# 请根据上下文示例的格式和已知事实三元组，从已知事实三元组中提取答案，简洁和专业的来回答用户的问题，答案请使用中文。
# 知识图谱"
# PROMPT_TEMPLATE = """
# 问题：{train_context_question1}
# 事实三元组：{train_context_input1}
# 答案：{train_context_answer1}
# ======
# 问题：{train_context_question2}
# 事实三元组：{train_context_input2}
# 答案：{train_context_answer2}
# ======
# 问题：{train_context_question3}
# 事实三元组：{train_context_input3}
# 答案：{train_context_answer3}
# ======
# 问题：{question}
# 事实三元组：{context}
# 答案："""

# 缓存知识库数量
CACHED_VS_NUM = 1

# 文本分句长度  #当第一步句子划分后，如果第一步后超过SENTENCE_SIZE，然后依据，号等进行加新一步划分
SENTENCE_SIZE = 200#匹配到的相似向量的长度  #在这里设置成什么都一样（因为都是一行了）

# 匹配后单段上下文长度（在匹配的文本分句长度前后扩充至CHUNK_SIZE字）
CHUNK_SIZE = 0 #这个参数决定了要不要后边的句子

# LLM input history length# LLM streaming reponse
LLM_HISTORY_LEN = 3 #表示LLM输入历史的长度

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 3 #表示从向量存储中返回的前k个文本块。默认为3。

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 500 #表示向量搜索的相关度阈值。默认为500。

#表示NLTK数据的路径。默认为当前文件的上一级目录下的"nltk_data"文件夹。
# os.path.dirname(__file__)：返回脚本的路径 ；__file__表示了当前文件的绝对路径 ；dirname表示去掉文件名，返回目录
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

#用于标记用户名。生成一个随机的唯一标识符。
FLAG_USER_NAME = uuid.uuid4().hex

logger.info(f"""
loading model config
llm device: {LLM_DEVICE}
embedding device: {EMBEDDING_DEVICE}
dir: {os.path.dirname(os.path.dirname(__file__))}
flagging username: {FLAG_USER_NAME}
""")#用于输出一些日志信息，包括LLM设备、嵌入式模型设备、NLTK数据路径等。

# 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = True

# Bing 搜索必备变量
# 使用 Bing 搜索需要使用 Bing Subscription Key
# 具体申请方式请见 https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
BING_SUBSCRIPTION_KEY = ""