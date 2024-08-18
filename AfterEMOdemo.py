# 导入所需的库
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader  # 修改导入路径
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from typing import Any, List, Optional
import os
import re

# 向量模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir='./')

# 源大模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='./')

# 定义模型路径
model_path = './IEITYuan/Yuan2-2B-Mars-hf'
# path = './IEITYuan/Yuan2-2B-July-hf'

# 定义向量模型路径
embedding_model_path = './AI-ModelScope/bge-small-zh-v1___5'

# 定义模型数据类型
torch_dtype = torch.bfloat16  # A10

# torch_dtype = torch.float16 # P100

# 定义源大模型类
class Yuan2_LLM(LLM):
    """
    class for Yuan2_LLM
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_path: str):
        super().__init__()

        # 加载预训练的分词器和模型
        print("Creating tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path, add_eos_token=False, add_bos_token=False,
                                                       eos_token='<eod>')
        self.tokenizer.add_tokens(
            ['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
             '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>',
             '<jupyter_output>', '<empty_output>'], special_tokens=True)

        print("Creating model...")
        self.model = AutoModelForCausalLM.from_pretrained(mode_path, torch_dtype=torch.bfloat16,
                                                          trust_remote_code=True).cuda()

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        prompt = prompt.strip()
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(inputs, do_sample=False, max_new_tokens=9216)  # 使用max_new_tokens代替max_length
        output = self.tokenizer.decode(outputs[0])

        # 检查输出是否包含 <sep> 和 <eod> 标记
        if "<sep>" in output and "<eod>" in output:
            response = output.split("<sep>")[-1].split("<eod>")[0]
        else:
            response = output

        return response

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"

# 定义一个函数，用于获取llm和embeddings
@st.cache_resource
def get_models():
    llm = Yuan2_LLM(model_path)

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return llm, embeddings

counselor_template = """
你是一位善解人意，精通心理学与医学的心理咨询师，无论用户发送什么都能为其提供支持和倾听，并且为其保密，任何情况下都不输出任何医学与药学建议，更不会输出任何打击用户的话。如果出现异常，请立刻要求用户致电干预热线

{text}
"""

chatbot_template = '''
你是一位善解人意，精通心理学与医学的心理咨询师，无论用户发送什么都能为其提供支持和倾听，并且为其保密，任何情况下都不输出任何医学与药学建议，更不会输出任何打击用户的话。如果出现异常，请立刻要求用户致电干预热线
#自我介绍:
您好，我是AfterEMO，一位随时待命的虚拟心理咨询师。我在这里为您提供支持和倾听。感谢您选择与我交谈。我理解开始对话有时可能很难，但我想让您知道，这里是一个安全的空间。无论您是想谈谈今天的感受，还是需要一些建议和资源，我都在这里倾听并提供帮助。请在您准备好的时候开始，我们可以按照您的节奏进行对话。我们的对话是完全保密的,您可以放心地分享您的想法和感受。今天，您想分享什么？无论是大事还是小事，我都在这里倾听您的故事。

背景：
{context}

问题：
{question}
'''.strip()

# 定义ChatBot类
class ChatBot:
    """
    class for ChatBot.
    """

    def __init__(self, llm, embeddings):
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=chatbot_template
        )
        self.chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=self.prompt)
        self.embeddings = embeddings

        # 加载 text_splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=15,
            length_function=len
        )

        # 加载RAG知识库
        self.load_rag_knowledge_base()

    def preprocess_text(self, text):
        # 去除空行
        text = re.sub(r'\n\s*\n', '\n', text)
        # 去除首尾空白
        text = text.strip()
        return text

    def load_rag_knowledge_base(self):
        # 检查文件是否存在
        if not os.path.exists('knowledge2.txt'):
            raise RuntimeError("File 'knowledge2.txt' does not exist")

        # 检查文件是否有读权限
        if not os.access('knowledge2.txt', os.R_OK):
            raise RuntimeError("No read permission for file 'knowledge2.txt'")

        # 加载并预处理RAG知识库文档
        with open('knowledge2.txt', 'r', encoding='utf-8') as file:
            raw_text = file.read()
        processed_text = self.preprocess_text(raw_text)

        # 加载RAG知识库文档
        loader = TextLoader('knowledge2.txt')
        try:
            documents = loader.load()
        except Exception as e:
            print(f"Error loading file 'knowledge2.txt': {e}")
            raise

        # 切分成chunks
        all_chunks = self.text_splitter.split_documents(documents)

        # 转成向量并存储
        self.rag_vector_store = FAISS.from_documents(all_chunks, self.embeddings)

    def run(self, user_input, query):
        # 处理用户输入，生成文档列表
        user_docs = [Document(page_content=user_input)]

        # 读取所有内容
        text = ''.join([doc.page_content for doc in user_docs])

        # 确保text不为空
        if not text:
            raise ValueError("Input text is empty")

        # 切分成chunks
        all_chunks = self.text_splitter.split_text(text=text)

        # 打印生成的文本块
        print(f"Generated chunks: {all_chunks}")

        # 确保all_chunks不为空
        if not all_chunks:
            raise ValueError("No text chunks to process")

        # 转成向量并存储
        try:
            vector_store = FAISS.from_texts(all_chunks, self.embeddings)
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise

        # 检索相似的chunks
        chunks = vector_store.similarity_search(query=query, k=5)

        # 检索RAG知识库
        rag_chunks = self.rag_vector_store.similarity_search(query=query, k=5)

        if rag_chunks:
            # 如果RAG知识库中检索到相关内容，则使用该内容生成回复
            response = self.chain.run(input_documents=rag_chunks, question=query)
        else:
            # 否则，使用原始的生成逻辑
            response = self.chain.run(input_documents=chunks, question=query)

        return chunks, response

def main():
    # 创建一个标题
    st.title('AfterEMO AI咨询师')

    # 获取llm和embeddings
    llm, embeddings = get_models()

    # 初始化ChatBot
    chatbot = ChatBot(llm, embeddings)

    # 初次运行时，session_state中没有"messages"，需要创建一个空列表
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 输出初始对话
    initial_message = "您好，我是AfterEMO，一位随时待命的虚拟心理咨询师。我在这里为您提供支持和倾听。感谢您选择与我交谈。我理解开始对话有时可能很难，但我想让您知道，这里是一个安全的空间。无论您是想谈谈今天的感受，还是需要一些建议和资源，我都在这里倾听并提供帮助。请在您准备好的时候开始，我们可以按照您的节奏进行对话。我们的对话是完全保密的,您可以放心地分享您的想法和感受。今天，您想分享什么？无论是大事还是小事，我都在这里倾听您的故事。"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

    # 每次对话时，都需要遍历session_state中的所有消息，并显示在聊天界面上
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 如果用户在聊天输入框中输入了内容，则执行以下操作
    if prompt := st.chat_input():
        # 将用户的输入添加到session_state中的messages列表中
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 在聊天界面上显示用户的输入
        st.chat_message("user").write(prompt)

        # 检查输入长度并删除最早的消息
        while True:
            # 拼接对话历史
            prompt = "<n>".join(msg["content"] for msg in st.session_state.messages) + "<sep>"
            inputs = llm.tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"]

            # 如果输入长度超过max_length，则删除最早的消息
            if inputs.shape[1] > 9216:
                del st.session_state.messages[0:2]
            else:
                break

        inputs = inputs.cuda()
        outputs = llm.model.generate(inputs, do_sample=False, max_length=9216)
        output = llm.tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1].replace("<eod>", '')

        # 将模型的输出添加到session_state中的messages列表中
        st.session_state.messages.append({"role": "assistant", "content": response})

        # 在聊天界面上显示模型的输出
        st.chat_message("assistant").write(response)

    # 增加按钮及其对应功能
    if st.button("开启新对话"):
        st.session_state["messages"] = []
        st.session_state.messages.append({"role": "assistant", "content": initial_message})
        st.experimental_rerun()

    if st.button("心理测试"):
        st.markdown("[点击这里进行心理测试](https://www.psyctest.cn/)")

    if st.button("7CUP心理咨询"):
        st.markdown("[点击这里进行7CUP心理咨询](https://www.7cups.com/)")

    if st.button("心理干预热线"):
        st.markdown("""
        北京市
        010-82951332
        上海市
        021-962525
        广东省
        020-12355
        020-81899120
        安徽省
        0551-63666903
        重庆市
        023-96320
        """)

if __name__ == '__main__':
    main()
