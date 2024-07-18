from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import(
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from utils import LOG, get_data_file_path
from sql.data_to_db import DataTODB
import os

class AlgorithmDescription:
    def __init__(self, model_name : str, api_key : str, base_url : str):
        try:    
            algorithm_describe_faiss_path = get_data_file_path("algorithms_describe")
            algorithm_describe_text_path = get_data_file_path("algorithm_describe")
            embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

            chatmodel = ChatOpenAI(
                model = model_name,
                api_key = api_key,
                base_url = base_url
            )
            self.chatmodel = chatmodel

            if os.path.exists(algorithm_describe_faiss_path):
                # 加载FAISS数据库
                db = FAISS.load_local(algorithm_describe_faiss_path, embeddings, allow_dangerous_deserialization=True)
            else:
                datatodb = DataTODB()
                db = datatodb.Describe_LocalToDB(algorithm_describe_text_path)
            self.db = db
        except Exception as e:
            LOG.error(f"Initialization failed: {e}")
            raise

    def GetAlgorithmChaoName(self, input : str, score_threshold : int):
        try:
            retriever = self.db.as_retriever(
                search_type = "similarity_score_threshold",
                search_kwargs = {"score_threshold" : score_threshold, "k" : 1},

            )
                
            system_template = (
                """
                你是国际信息学奥林匹克竞赛冠军,并且也是国际大学生程序设计大赛ICPC-WF冠军。
                以检索到的内容中'[回答]'的下一行中()中的内容作为你的输出!!
                使用以下检索到的内容来回答问题。
                {context}
                """
            )
            system_prompt = SystemMessagePromptTemplate.from_template(system_template)

            human_template = ("{input}")
            human_prompt1 = HumanMessagePromptTemplate.from_template(human_template)

            prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt1])

            chain = create_stuff_documents_chain(self.chatmodel, prompt)

            retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=chain)
            result = retrieval_chain.invoke({'input' : input})
            LOG.debug(f"[context]{result['context']}")
            return result['answer'], True
        except Exception as e:
            LOG.error(f"Retrieval failed: {e}")
            return "An error occurred during retrieval", False

        

        