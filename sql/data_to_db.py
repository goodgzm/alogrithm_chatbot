from sql.mysql import SQLite
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter 
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from utils import LOG
import re

class DataTODB:
    def __init__(self):
        sqlite = SQLite()
        self.sqlite = sqlite
    
    def Name_And_Code_LocalToDB(self, algorithm_code_path : str):
        loader = TextLoader(algorithm_code_path, encoding="UTF-8")
        document = loader.load()

        text_splitter = CharacterTextSplitter(
                separator = r'\n\n-----\d+-----',
                chunk_size=300,
                chunk_overlap=0,
                length_function = len,
                is_separator_regex = True
            )
        
        docs = text_splitter.split_documents(document)
        for doc in docs:
            text = doc.page_content
            lines = text.strip().split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if line == "[算法名称]":
                    current_section = "算法名称"
                    algorithm_name = ""
                elif line == "[算法代码]":
                    current_section = "算法代码"
                    algorithm_code = ""
                elif current_section == "算法名称":
                    algorithm_name += line + "\n"
                elif current_section == "算法代码":
                    algorithm_code += line + "\n"

            algorithm_name = algorithm_name.strip()
            algorithm_code = algorithm_code.strip()
            LOG.debug(f'[algorithm_name]{algorithm_name}')
            LOG.debug(f'[algorithm_code]{algorithm_code}')
            statue = True
            statue |= self.sqlite.Insert_Data_name_and_id(algorithm_name, algorithm_code)
        return statue
    

    def Result_Online_ToDB_By_ID(self, algorithm_id : int, new_result : str):
        return self.sqlite.Insert_Date_result_By_ID(algorithm_id, new_result)
    

    def Describe_LocalToDB(self, vector_store_path : str):
        loader = TextLoader(vector_store_path, encoding="UTF-8")

        document = loader.load()
        
        text_splitter = CharacterTextSplitter(
            separator = r'\n\n-----\d+-----',
            chunk_size=300,
            chunk_overlap=0,
            length_function = len,
            is_separator_regex = True
        )

        docs = text_splitter.split_documents(document)
        

        db = FAISS.from_documents(docs, HuggingFaceBgeEmbeddings(
            model_name = "BAAI/bge-small-zh-v1.5",
            model_kwargs = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True}
        )) 

        db.save_local("D://jupyter//alogrithm_chatbot//algorithms_describe")  

        return db