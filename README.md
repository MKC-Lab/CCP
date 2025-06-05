# CCP 本地文档问答与知识检索系统

本项目实现了基于本地文档的智能问答系统，支持文档向量化检索、知识库管理、BLEU/BERTScore/ROUGE 等多种评测方式，并集成了命令行与 Web 界面。适用于中文语境下的知识问答、文档检索和智能对话。

## 目录结构

```plaintext
├── api.py                      # FastAPI 接口服务
├── BertScore.py                # BERTScore 评测脚本
├── Chatzhuanzhi.py             # Gradio Web UI 主程序
├── cli_demo.py                 # 命令行交互 demo
├── cli.py                      # 命令行主入口
├── config_utils.py             # 配置工具
├── dataset.py                  # 数据集处理
├── evaluate.py                 # BLEU/ROUGE 评测脚本
├── inference_chatzhuanzhi.py   # 推理脚本
├── local_doc_qa.py             # 本地文档问答主逻辑
├── model_config.py             # 模型与参数配置
├── release.py                  # 版本发布工具
├── requirements.txt            # 依赖列表
├── search.py                   # 检索与问答主程序
├── agent/                      # Agent 相关代码
├── chains/                     # 链式问答逻辑
├── chatzhuanzhi/               # 专用知识库与 DPR 相关代码
├── configs/                    # 配置文件
├── data/                       # 数据集与样例数据
├── loader/                     # 模型加载与工具
├── models/                     # 语言模型与嵌入模型
├── retrieve/                   # 检索相关代码
├── test/                       # 测试脚本
├── textsplitter/               # 文本分割工具
├── utils/                      # 工具函数
```


## 目录结构

### 1. 环境准备

- Python 3.8+
- 推荐使用虚拟环境管理项目依赖
- 安装依赖包：

```bash
pip install -r requirements.txt
```

下载 NLTK 数据（如未自动下载）
```python
import nltk
nltk.download('punkt')
```

### 2.启动命令行 Demo 
```python
python [cli_demo.py](http://_vscodecontentref_/15)
```

### 3.启动 Web UI
```python
python [Chatzhuanzhi.py](http://_vscodecontentref_/17)
```

### 4. 启动 API 服务
```python
python [api.py](http://_vscodecontentref_/18)
```
提供基于 FastAPI 的 RESTful 接口和 WebSocket 实时问答接口。

## 主要功能
本地文档知识库管理：支持 txt/docx/pdf 等格式文档的导入、分割、向量化和检索。
多模型支持：可配置不同的 LLM（如 ChatGLM、Baichuan）和嵌入模型（如 text2vec）。
知识增强问答：结合检索与生成，支持上下文增强、历史多轮对话。
多种评测指标：内置 BLEU、ROUGE、BERTScore 等自动化评测脚本。
多端交互：支持命令行、Web UI、API 三种方式。

## 配置说明
所有模型与参数配置均在 model_config.py 中设置。
支持自定义知识库路径、模型路径、检索参数等。

## 评测脚本
BLEU/ROUGE 评测：evaluate.py
BERTScore 评测：BertScore.py

## 主要模块说明
chains/local_doc_qa.py：本地文档问答核心逻辑，包含知识库初始化、检索、问答等方法。
Chatzhuanzhi.py：Web UI 主入口，集成 Gradio。
api.py：API 服务主入口，集成 FastAPI。
search.py：命令行检索与问答主程序。

## 数据与知识库
默认知识库路径为 data/knowledge_data/，可根据实际需求替换。
支持多文档、多格式批量导入。